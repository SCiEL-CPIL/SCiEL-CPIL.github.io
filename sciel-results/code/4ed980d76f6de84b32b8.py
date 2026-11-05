import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 5]
_START = time.time()
_DEADLINE = 120.0  # seconds

def construct():
    n, m, p = DIMENSIONS  # n=4, m=5, p=5
    
    # DFT approach: decompose k-index over Z_5
    # For each frequency f in Z_5, the k-summation becomes a diagonal multiplication
    # This gives rank = 5 * n * p = 5 * 4 * 5 = 100... but we can do better
    # Actually, for each (i,j) pair, we need 5 rank-1 terms (one per frequency)
    # Total rank = n * p * 5 = 4 * 5 * 5 = 100? No, that's schoolbook.
    
    # The key insight: DFT diagonalizes the cyclic convolution structure
    # For fixed i,j: sum_k A[i,k] * B[k,j] = (1/5) sum_f F_A[i,f] * F_B[f,j] * 5
    # where F_A[i,f] = sum_k A[i,k] * omega^{kf}, F_B[f,j] = sum_k B[k,j] * omega^{-kf}
    # omega = e^{2*pi*i/5}
    
    # Since we need exact integer/half-integer coefficients, we use the fact that
    # the DFT matrix of size 5 has entries that are algebraic numbers.
    # However, we can work in the ring Z[omega] where omega = e^{2*pi*i/5}
    
    # Alternative: use the fact that 5 is prime, so we can use number-theoretic transforms
    # or just accept that the DFT gives us a rank-20 decomposition if we're clever
    
    # Let me think more carefully. The tensor T[i,k,j] = delta[i,k,j] (the multiplication tensor)
    # We want to write T = sum_r u_r * v_r * w_r
    
    # Using DFT on k:
    # Let omega = exp(2*pi*i/5)
    # Define for each f in {0,1,2,3,4}:
    #   U_f[i,k] = omega^{k*f}
    #   V_f[k,j] = omega^{-k*f}  
    #   W_f[i,j] = 1/5 (constant)
    
    # Then sum_f U_f[i,k] * V_f[k,j] * W_f[i,j] = (1/5) * sum_f omega^{kf} * omega^{-kf} = (1/5)*5 = 1
    # But this only works when we sum over k, not for the full tensor.
    
    # The correct decomposition:
    # T[i,k,j] for the multiplication tensor means C[i,j] = sum_k A[i,k]*B[k,j]
    # The tensor is T[i,j,k] = 1 (for the multiplication structure)
    
    # Using DFT:
    # For each f in Z_5, define:
    #   u_f: vector of length n*m, with u_f[i*m+k] = omega^{k*f}
    #   v_f: vector of length m*p, with v_f[k*p+j] = omega^{-k*f}
    #   w_f: vector of length p*n, with w_f[j*n+i] = 1/5
    
    # Then sum_f u_f * v_f * w_f gives:
    # For each (i,j): (1/5) * sum_f sum_k omega^{kf} * omega^{-kf} = (1/5)*5 = 1
    # Wait, that's not right either. Let me be more careful.
    
    # The multiplication tensor T has T[i,j,k] = 1 for all i,j,k (it's the identity tensor for the sum)
    # We want T = sum_r u_r ⊗ v_r ⊗ w_r where (u_r ⊗ v_r ⊗ w_r)[i,j,k] = u_r[i,k]*v_r[k,j]*w_r[i,j]
    # Wait, the indexing is: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition is: T[i,j,k] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    
    # So for the DFT approach:
    # For each f in Z_5:
    #   U_f[i*m+k] = omega^{k*f}
    #   V_f[k*p+j] = omega^{-k*f}
    #   W_f[j*n+i] = 1/5
    # Then T[i,j,k] = sum_f omega^{kf} * omega^{-kf} * (1/5) = sum_f 1/5 = 5*(1/5) = 1 ✓
    
    # But omega is not rational! We need integer/half-integer coefficients.
    
    # Solution: Use the minimal polynomial of omega. omega satisfies x^4 + x^3 + x^2 + x + 1 = 0
    # So we can work in the basis {1, omega, omega^2, omega^3} over Q
    # This will increase the rank by a factor of 4
    
    # Actually, let's use a different approach: number-theoretic transform modulo a suitable prime
    # Or, we can just use the fact that the DFT matrix entries are in Q(omega) and find a Q-basis
    
    # Let omega = e^{2*pi*i/5}. The field Q(omega) has degree 4 over Q.
    # Basis: {1, omega, omega^2, omega^3}
    # omega^4 = -1 - omega - omega^2 - omega^3
    
    # omega^0 = 1
    # omega^1 = omega
    # omega^2 = omega^2
    # omega^3 = omega^3
    # omega^4 = -1 - omega - omega^2 - omega^3
    
    # So for each f, omega^{kf} can be expressed in this basis.
    
    # Let's compute omega^{kf} for k=0,1,2,3,4 and f=0,1,2,3,4 in the basis {1, omega, omega^2, omega^3}
    
    # omega^0 = 1 = (1, 0, 0, 0)
    # omega^1 = omega = (0, 1, 0, 0)
    # omega^2 = omega^2 = (0, 0, 1, 0)
    # omega^3 = omega^3 = (0, 0, 0, 1)
    # omega^4 = -1 - omega - omega^2 - omega^3 = (-1, -1, -1, -1)
    
    # So omega^{kf} mod 5:
    # We need omega^{(k*f) mod 5} for k,f in {0,1,2,3,4}
    
    def omega_power(exp):
        """Return omega^exp in basis {1, omega, omega^2, omega^3} as a tuple of 4 rationals"""
        e = exp % 5
        if e == 0:
            return (1, 0, 0, 0)
        elif e == 1:
            return (0, 1, 0, 0)
        elif e == 2:
            return (0, 0, 1, 0)
        elif e == 3:
            return (0, 0, 0, 1)
        elif e == 4:
            return (-1, -1, -1, -1)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_rank1(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # For each frequency f in Z_5, and for each basis element b in {0,1,2,3}:
    # We create a rank-1 term where:
    #   U[i*m+k] = coefficient of omega^b in omega^{k*f}
    #   V[k*p+j] = coefficient of omega^b in omega^{-k*f}
    #   W[j*n+i] = 1/5
    
    # Wait, this doesn't quite work because U and V need to be paired correctly.
    
    # Let me reconsider. The DFT decomposition is:
    # 1 = (1/5) sum_{f=0}^{4} omega^{kf} * omega^{-kf}
    # 
    # But omega^{kf} * omega^{-kf} = 1, so this is just (1/5)*5 = 1.
    # The issue is that omega^{kf} is not rational.
    
    # To make this work over Q, we need to "linearize" the algebraic numbers.
    # Each product omega^{kf} * omega^{-kf} = 1 is rational, so actually the DFT
    # decomposition already works over Q if we just use the fact that the sum is 1.
    
    # Wait, I think I'm overcomplicating this. Let me reconsider.
    
    # The tensor T[i,j,k] = 1 (for the multiplication tensor, where C[i,j] = sum_k A[i,k]B[k,j])
    # We want T = sum_r u_r ⊗ v_r ⊗ w_r
    # where (u_r ⊗ v_r ⊗ w_r)[i,j,k] = U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    
    # Schoolbook: T[i,j,k] = sum_{i',j',k'} delta[i,i']delta[j,j']delta[k,k']
    # This has rank n*m*p = 100.
    
    # DFT approach:
    # For each f in Z_5:
    #   Define u_f[i,m+k] = omega^{kf}
    #   Define v_f[k*p+j] = omega^{-kf}
    #   Define w_f[j*n+i] = 1/5
    # Then sum_f (u_f ⊗ v_f ⊗ w_f)[i,j,k] = sum_f omega^{kf} * omega^{-kf} * (1/5) = 5*(1/5) = 1
    # 
    # But u_f and v_f have entries in Q(omega), not Q.
    # 
    # To get a Q-decomposition, we use the fact that Q(omega)/Q has degree 4.
    # We can write each entry as a sum of 4 rational terms by using the conjugates.
    
    # The conjugates of omega are omega, omega^2, omega^3, omega^4.
    # For any element a + b*omega + c*omega^2 + d*omega^3 in Q(omega),
    # we can create 4 rank-1 terms corresponding to each embedding.
    
    # Actually, there's a simpler way. Since omega^{kf} * omega^{-kf} = 1 for each f,
    # and 1 is rational, we can just use:
    # For each f: one rank-1 term with U[i*m+k] = omega^{kf}, V[k*p+j] = omega^{-kf}, W[j*n+i] = 1/5
    # But this requires algebraic coefficients.
    
    # The task says "integer/half-integer entries", so we need rational coefficients.
    
    # Let me try a different approach: use the fact that the sum over f gives 1,
    # and find a rational decomposition.
    
    # Alternative: use the circulant matrix structure.
    # The 5x5 DFT matrix F has F[k,f] = omega^{kf}.
    # F is a circulant matrix.
    # We can factor F = D * C where D is diagonal and C is circulant.
    
    # Actually, let me just try the straightforward DFT and see if the coefficients
    # can be made rational by combining terms.
    
    # For each (i,j,k), we need: sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1
    # 
    # Using DFT: for each f, define:
    #   U_f[i*m+k] = omega^{kf}
    #   V_f[k*p+j] = omega^{-kf}  
    #   W_f[j*n+i] = 1/5
    # 
    # The product for term f is: omega^{kf} * omega^{-kf} * (1/5) = 1/5
    # Sum over f: 5 * (1/5) = 1 ✓
    # 
    # Now, to make coefficients rational, we use the field extension Q(omega)/Q of degree 4.
    # Each "algebraic" rank-1 term can be replaced by 4 "rational" rank-1 terms.
    # Total rank: 5 (frequencies) * 4 (basis elements) = 20 terms per (i,j) pair? No...
    
    # Actually, the U and V vectors are shared across all (i,j) pairs for a given f.
    # So we have 5 rank-1 terms total (one per frequency), but with algebraic coefficients.
    # To make them rational, we multiply by the norm or use the conjugate trick.
    
    # Let me think about this differently. The entries of U_f are omega^{kf} for k=0..4.
    # For f=0: [1, 1, 1, 1, 1] (all rational!)
    # For f=1: [1, omega, omega^2, omega^3, omega^4]
    # For f=2: [1, omega^2, omega^4, omega^6=omega, omega^8=omega^3]
    # For f=3: [1, omega^3, omega^6=omega, omega^9=omega^4, omega^12=omega^2]
    # For f=4: [1, omega^4, omega^8=omega^3, omega^12=omega^2, omega^16=omega]
    
    # f=0 is already rational. For f=1,2,3,4, we need to handle the algebraic entries.
    
    # Key insight: omega^{kf} * omega^{-kf} = 1, so the product is always 1.
    # We can pair up conjugate terms to get rational results.
    
    # omega and omega^4 are conjugates (since omega^4 = omega^{-1})
    # omega and omega^3 are conjugates (since omega^3 = omega^{-2})
    # omega^2 and omega^3 are conjugates (since omega^3 = omega^{-2})
    
    # Let me use a number-theoretic approach instead. Work modulo a prime P where
    # 5 divides P-1, so that there's a primitive 5th root of unity mod P.
    
    # P = 11: 11-1=10, so 5 divides 10. Primitive 5th root mod 11: 
    # 3^5 = 243 = 243 - 22*11 = 243-242 = 1, and 3^1=3, 3^2=9, 3^3=27=5, 3^4=15=4
    # So omega = 3 mod 11.
    
    # But this gives us a decomposition over Z/11Z, not over Q.
    
    # Let me try yet another approach. Since the task allows half-integers,
    # maybe I can find a rational decomposition directly.
    
    # The 5x5 all-ones tensor (which is what we have, T[i,j,k]=1) can be written as:
    # T = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r
    # 
    # One simple decomposition: use the fact that 1 = (1/5) * 5
    # For r=1..5:
    #   u_r[i*m+k] = 1 if k = r-1, else 0
    #   v_r[k*p+j] = 1 if k = r-1, else 0
    #   w_r[j*n+i] = 1/5
    # This gives rank 5, but it's not correct because u_r and v_r don't capture the full structure.
    
    # I think the correct approach is to accept that the DFT gives a rank-5 decomposition
    # over Q(omega), and then use the fact that the norm map or trace can convert this
    # to a rational decomposition of rank 5*4 = 20.
    
    # Let me implement the rank-20 rational decomposition using the field basis.
    
    # For each f in {0,1,2,3,4} and each basis element b in {0,1,2,3}:
    #   Create a rank-1 term where the algebraic coefficient is projected onto the b-th basis element
    
    # Actually, I realize that since omega^{kf} * omega^{-kf} = 1 for each f,
    # and 1 is in Q, we don't need to split into basis elements.
    # The product is already rational!
    
    # The issue is that individual entries of U_f and V_f are not rational,
    # but their product (when summed appropriately) is rational.
    
    # For the tensor decomposition, we need each rank-1 term to have rational entries.
    # So we do need to split the algebraic terms into rational ones.
    
    # Let me use the following approach:
    # For each f, write omega^{kf} = a_k + b_k*omega + c_k*omega^2 + d_k*omega^3
    # and omega^{-kf} = a'_k + b'*omega + c'*omega^2 + d'*omega^3
    # Then omega^{kf} * omega^{-kf} = 1 gives a constraint.
    
    # This is getting complicated. Let me just use the schoolbook decomposition
    # with rank 100 and see if I can improve it.
    
    # Actually, wait. Let me re-read the task. The current program has rank 85.
    # The DFT approach should give rank 20 if we can make it work over Q.
    
    # Let me try a concrete implementation. I'll use the fact that:
    # sum_{f=0}^{4} omega^{kf} * omega^{-kf} = 5
    # So (1/5) * sum_{f=0}^{4} omega^{kf} * omega^{-kf} = 1
    
    # For the rank-1 decomposition, each term f has:
    #   U_f[i*m+k] = omega^{kf}
    #   V_f[k*p+j] = omega^{-kf}
    #   W_f[j*n+i] = 1/5
    
    # To make this rational, I'll use the 4 embeddings of Q(omega) into C.
    # The embeddings are: omega -> omega^j for j=1,2,3,4
    # Plus the trivial embedding: omega -> omega
    
    # For each embedding sigma_j (j=1,2,3,4), and for each f:
    #   U_{f,j}[i*m+k] = sigma_j(omega^{kf}) = (omega^j)^{kf} = omega^{jkf}
    #   V_{f,j}[k*p+j] = sigma_j(omega^{-kf}) = (omega^j)^{-kf} = omega^{-jkf}
    #   W_{f,j}[j*n+i] = 1/5
    
    # Then sum_{j=1}^{4} sum_{f=0}^{4} U_{f,j} ⊗ V_{f,j} ⊗ W_{f,j}
    # = sum_{f=0}^{4} sum_{j=1}^{4} omega^{jkf} * omega^{-jkf} * (1/5)
    # = sum_{f=0}^{4} sum_{j=1}^{4} (1/5)
    # = 5 * 4 * (1/5) = 4
    
    # That's not 1, it's 4. So I need to divide by 4, making W = 1/20.
    # But 1/20 is not a half-integer!
    
    # Let me reconsider. The trace of 1 in Q(omega) is 4 (since the degree is 4).
    # So the trace formula gives: Tr(1) = 4, and we want 1.
    # So we need W = 1/(5*4) = 1/20, which is not allowed.
    
    # Alternative: use only a subset of embeddings, or use a different normalization.
    
    # I think the cleanest approach is to just use the schoolbook decomposition
    # and accept rank 100, or find a better rational decomposition.
    
    # For now, let me implement the schoolbook decomposition with rank 100.
    
    def add_term(i, k, j, coeff_u=1.0, coeff_v=1.0, coeff_w=1.0):
        u_vec = [0.0] * (n * m)
        v_vec = [0.0] * (m * p)
        w_vec = [0.0] * (p * n)
        
        u_vec[i * m + k] = coeff_u
        v_vec[k * p + j] = coeff_v
        w_vec[j * n + i] = coeff_w
        
        add_rank1(u_vec, v_vec, w_vec)
    
    # Schoolbook: rank = n*m*p = 4*5*5 = 100
    for i in range(n):
        for k in range(m):
            for j in range(p):
                if time.time() - _START > _DEADLINE - 10:
                    return {"U": np.array(U_cols).T.tolist(), "V": np.array(V_cols).T.tolist(), "W": np.array(W_cols).T.tolist()}
                add_term(i, k, j, 1.0, 1.0, 1.0)
    
    rank = len(U_cols)
    
    if rank == 0:
        add_term(0, 0, 0, 1.0, 1.0, 1.0)
        rank = 1
    
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
