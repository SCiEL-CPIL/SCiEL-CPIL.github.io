import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
import time

# Global cache for Hamiltonian parts to avoid rebuilding if called multiple times in same process?
# The prompt says "coefficients runs in a sandboxed subprocess", so likely one call per eval.
# But we can build it inside.

def build_hamiltonians(L=6, Delta=0.8, h=0.3):
    """
    Builds A and B matrices for the 6-site open XXZ Heisenberg chain.
    A: Odd bonds (1-2, 3-4, 5-6) + half field
    B: Even bonds (2-3, 4-5) + half field
    Note: Bond indices are 1-based in description, 0-based in code.
    Bonds: (0,1), (1,2), (2,3), (3,4), (4,5).
    Odd bonds (1st, 3rd, 5th): (0,1), (2,3), (4,5).
    Even bonds (2nd, 4th): (1,2), (3,4).
    """
    dim = 2 ** L
    # Pauli matrices
    sx = 0.5 * np.array([[0, 1], [1, 0]])
    sy = 0.5 * np.array([[0, -1j], [1j, 0]])
    sz = 0.5 * np.array([[1, 0], [0, -1]])
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Helper to add term S_i^alpha S_j^beta
    def add_term(mat, i, j, op_i, op_j):
        # Create operator on site i and j
        # Identity on all other sites
        ops = [np.eye(2)] * L
        ops[i] = op_i
        ops[j] = op_j
        # Tensor product
        term = ops[0]
        for k in range(1, L):
            term = np.kron(term, ops[k])
        mat += term

    # Bonds
    bonds = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    
    for idx, (i, j) in enumerate(bonds):
        # Sx Sx + Sy Sy + Delta Sz Sz
        term_val = add_term_and_return(A if idx % 2 == 0 else B, i, j, sx, sx) + \
                    add_term_and_return(A if idx % 2 == 0 else B, i, j, sy, sy) + \
                    Delta * add_term_and_return(A if idx % 2 == 0 else B, i, j, sz, sz)
        
    # Field
    # H_field = h * sum S^z
    # Split: h/2 * sum S^z in A, h/2 * sum S^z in B
    for site in range(L):
        add_term_and_return(A, site, site, sz, None) # This helper needs fixing for single site
        # Actually, let's just do it manually or fix helper.
        
    # Let's rewrite add_term to be cleaner
    return A, B

def add_term_and_return(mat, i, j, op_i, op_j):
    dim = 2**6
    ops = [np.eye(2)] * 6
    ops[i] = op_i
    if j is not None and j != i:
        ops[j] = op_j
    term = ops[0]
    for k in range(1, 6):
        term = np.kron(term, ops[k])
    mat += term
    return 1.0 # dummy return

# Corrected build function
def get_AB():
    L = 6
    dim = 2 ** L
    sx = 0.5 * np.array([[0, 1], [1, 0]])
    sy = 0.5 * np.array([[0, -1j], [1j, 0]])
    sz = 0.5 * np.array([[1, 0], [0, -1]])
    Delta = 0.8
    h = 0.3
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    def add_bond(mat, i, j):
        # Sx Sx
        ops = [np.eye(2)] * L
        ops[i] = sx
        ops[j] = sx
        term = ops[0]
        for k in range(1, L): term = np.kron(term, ops[k])
        mat += term
        
        # Sy Sy
        ops[i] = sy
        ops[j] = sy
        term = ops[0]
        for k in range(1, L): term = np.kron(term, ops[k])
        mat += term
        
        # Sz Sz * Delta
        ops[i] = sz
        ops[j] = sz
        term = ops[0]
        for k in range(1, L): term = np.kron(term, ops[k])
        mat += Delta * term

    def add_field(mat, coeff):
        for site in range(L):
            ops = [np.eye(2)] * L
            ops[site] = sz
            term = ops[0]
            for k in range(1, L): term = np.kron(term, ops[k])
            mat += coeff * term

    # Bonds: (0,1) odd->A, (1,2) even->B, (2,3) odd->A, (3,4) even->B, (4,5) odd->A
    add_bond(A, 0, 1)
    add_bond(B, 1, 2)
    add_bond(A, 2, 3)
    add_bond(B, 3, 4)
    add_bond(A, 4, 5)
    
    add_field(A, h / 2.0)
    add_field(B, h / 2.0)
    
    return A, B

def coefficients(stages):
    """
    Optimizes coefficients for a 12-stage product formula.
    """
    if stages != 12:
        # Fallback for other stage counts if needed, but task specifies 12
        # We can scale Suzuki 4th order
        substeps = max(1, stages // 4) # rough
        # Just return zeros or basic Suzuki for safety
        a = [0.0] * stages
        b = [0.0] * stages
        # Simple 2nd order Strang: A(0.5) B(1) A(0.5)
        # This doesn't fit 12 easily. 
        # For now, assume 12.
        return a, b

    A, B = get_AB()
    H = A + B
    
    # Exact propagators
    t_vals = [0.25, 0.5, 1.0]
    U_exact = [la.expm(-1j * t * H) for t in t_vals]
    
    # Initial guess: 4th order Suzuki 3 substeps
    # c1 = 1 / (2 - 2**(1/3))
    # c2 = -2**(1/3) / (2 - 2**(1/3))
    c1 = 1.0 / (2.0 - 2.0**(1.0/3.0))
    c2 = -2.0**(1.0/3.0) / (2.0 - 2.0**(1.0/3.0))
    
    # One 4th order step: B(c1/2) A(c1) B((c1+c2)/2) A(c2) B(c1/2)
    # 3 substeps of t/3.
    # We need to map this to 12 pairs (B, A).
    # The sequence is B A B A B B A B A B B A B A B
    # Merged B's:
    # 1. B(c1/2)
    # 2. A(c1)
    # 3. B((c1+c2)/2 + c1/2) = B(c1 + c2/2)
    # 4. A(c2)
    # 5. B((c1+c2)/2)
    # 6. A(c1)
    # 7. B(c1/2 + (c1+c2)/2) = B(c1 + c2/2)
    # 8. A(c2)
    # 9. B((c1+c2)/2)
    # 10. A(c1)
    # 11. B(c1/2)
    
    # We have 12 slots for B and 12 for A.
    # We can distribute these 6 B's and 5 A's into 12 slots.
    # To maintain the structure, we can just put them in order and pad with 0s.
    # But the structure is strictly B_1 A_1 B_2 A_2 ... B_12 A_12.
    # The Suzuki sequence ends with B. Our format ends with A.
    # So we can shift:
    # Slot 1: B(c1/2), A(0)
    # Slot 2: B(0), A(c1)
    # Slot 3: B(c1+c2/2), A(0)
    # Slot 4: B(0), A(c2)
    # Slot 5: B((c1+c2)/2), A(0)
    # Slot 6: B(0), A(c1)
    # Slot 7: B(c1+c2/2), A(0)
    # Slot 8: B(0), A(c2)
    # Slot 9: B((c1+c2)/2), A(0)
    # Slot 10: B(0), A(c1)
    # Slot 11: B(c1/2), A(0)
    # Slot 12: B(0), A(0)
    
    # This uses 11 slots. We have 12.
    # We can just use this as initial guess for optimization.
    
    b_init = [0.0] * 12
    a_init = [0.0] * 12
    
    # Scale by 1/3 for each substep
    s = 1.0 / 3.0
    
    b_init[0] = c1 * s / 2.0
    a_init[1] = c1 * s
    b_init[2] = (c1 + c2/2.0) * s
    a_init[3] = c2 * s
    b_init[4] = (c1 + c2)/2.0 * s
    a_init[5] = c1 * s
    b_init[6] = (c1 + c2/2.0) * s
    a_init[7] = c2 * s
    b_init[8] = (c1 + c2)/2.0 * s
    a_init[9] = c1 * s
    b_init[10] = c1 * s / 2.0
    
    # Normalize to ensure sums are 1 (they should be close)
    sum_a = sum(a_init)
    sum_b = sum(b_init)
    if sum_a != 0:
        a_init = [x / sum_a for x in a_init]
    if sum_b != 0:
        b_init = [x / sum_b for x in b_init]
        
    # Objective function
    def objective(x):
        a = x[:12]
        b = x[12:]
        
        # Ensure constraints are met? 
        # We will optimize 22 variables and constrain the last two.
        # Or just optimize 24 and add penalty.
        # Better: optimize 22, set a_11 = 1 - sum(a_0..10), b_11 = 1 - sum(b_0..10)
        
        a_full = list(a[:11]) + [1.0 - sum(a[:11])]
        b_full = list(b[:11]) + [1.0 - sum(b[:11])]
        
        err = 0.0
        for t, Ue in zip(t_vals, U_exact):
            # Build U(t)
            U = np.eye(dim, dtype=complex)
            # Order: prod k=1..12 exp(-i b_k t B) exp(-i a_k t A)
            # This is right-to-left multiplication? 
            # "prod_k ... in order k=1..12" usually means U = U_12 U_11 ... U_1
            # Where U_k = exp(-i b_k t B) exp(-i a_k t A)
            # So U = U_12 * U_11 * ... * U_1
            for k in range(11, -1, -1):
                U = la.expm(-1j * b_full[k] * t * B) @ la.expm(-1j * a_full[k] * t * A) @ U
            
            err += la.norm(U - Ue, 2)
        
        return err / len(t_vals)
    
    # Optimization
    x0 = b_init[:11] + a_init[:11]
    
    start_time = time.time()
    timeout = 45.0 # Leave margin
    
    best_x = x0
    best_val = objective(x0)
    
    # Use Nelder-Mead
    try:
        res = opt.minimize(
            objective, 
            x0, 
            method='Nelder-Mead', 
            options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5, 'disp': False}
        )
        
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
            
    except Exception:
        pass
        
    # Check time
    if time.time() - start_time > timeout:
        # Fallback to initial guess if optimization failed or timed out
        best_x = x0
        best_val = objective(x0)

    b_opt = best_x[:11]
    a_opt = best_x[11:]
    
    b_final = list(b_opt) + [1.0 - sum(b_opt)]
    a_final = list(a_opt) + [1.0 - sum(a_opt)]
    
    # Clamp to [-50, 50]
    b_final = [max(-50.0, min(50.0, x)) for x in b_final]
    a_final = [max(-50.0, min(50.0, x)) for x in a_final]
    
    # Re-normalize after clamping
    sum_b = sum(b_final)
    sum_a = sum(a_final)
    if sum_b != 0:
        b_final = [x / sum_b for x in b_final]
    if sum_a != 0:
        a_final = [x / sum_a for x in a_final]
        
    return a_final, b_final

