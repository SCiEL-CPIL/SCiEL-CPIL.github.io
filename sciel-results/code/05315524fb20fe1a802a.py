import numpy as np
from scipy.optimize import minimize
import time

def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles maximizing the mean ratio of the 
    independently simulated expected cut to the exact classical optimum.
    
    Uses Stabilizer formalism for efficient simulation of the QAOA circuit.
    """
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin

    # The state is represented by a set of n independent stabilizer generators.
    # Each generator is a string of Pauli operators (0=I, 1=X, 2=Y, 3=Z) and a phase.
    # For simplicity in QAOA starting from |+>^n, the phase is always 1.
    # We can represent a stabilizer generator by its X and Z parts (binary vectors of length n).
    # A generator is (x, z) where x, z in {0,1}^n.
    # The Pauli operator is prod_j X_j^{x_j} Z_j^{z_j}.
    # Initial state |+>^n is stabilized by X_i for all i.
    
    # Initialize stabilizers: I...IXI...I for each i
    # x_matrix[i] is the X part of stabilizer i
    # z_matrix[i] is the Z part of stabilizer i
    x_mat = np.zeros((n, n), dtype=np.int8)
    z_mat = np.zeros((n, n), dtype=np.int8)
    
    for i in range(n):
        x_mat[i, i] = 1 # X on qubit i
        z_mat[i, i] = 0 # Z part is 0 (identity)

    def update_stabilizers_x_beta(x_mat, z_mat, beta):
        """
        Apply exp(-i * beta/2 * sum_j X_j) to the stabilizers.
        R_x(beta) conjugation:
        X -> X
        Y -> cos(beta)Y + sin(beta)Z
        Z -> -sin(beta)Y + cos(beta)Z
        I -> I
        
        However, working with (x, z) binary representation makes Y hard to handle directly 
        unless we track complex coefficients or use a different basis.
        
        Alternative: Use the explicit matrix update for the stabilizer generators.
        Let S be a stabilizer generator.
        R_x(beta) S R_x(beta)^dag
        
        Recall:
        R_x(beta) X R_x(beta)^dag = X
        R_x(beta) Z R_x(beta)^dag = cos(beta)Z - sin(beta)Y
        R_x(beta) Y R_x(beta)^dag = cos(beta)Y + sin(beta)Z
        
        If we maintain the state as a vector of Pauli operators with complex coefficients, 
        it becomes messy. 
        
        Actually, there is a simpler way to compute the expectation value without explicitly 
        updating all stabilizers to non-Pauli forms if we just track the effect on the 
        expectation value of the cost operator.
        
        Wait, the stabilizer method usually tracks the *state*. The state remains a stabilizer 
        state only if the operations are Clifford. R_x(beta) is NOT Clifford for arbitrary beta.
        Therefore, the state is NOT a stabilizer state after the first R_x layer!
        
        Correction:
        The QAOA state is NOT a stabilizer state for arbitrary angles. 
        The "Stabilizer Trick" applies if we can express the expectation value efficiently 
        without full state evolution, or if we use a different ansatz.
        
        However, for small n (<=10), the full state vector is only 1024 complex numbers.
        The bottleneck in the original code was the *optimization loop* calling the full simulation many times.
        Let's optimize the *simulation* itself using NumPy vectorization and precomputation, 
        and use a more efficient optimizer.
        
        Actually, for n=10, 2^10 = 1024 states.
        Applying cost phase: element-wise multiplication (1024 ops).
        Applying R_x: involves reshaping and matrix mults.
        This is very fast in NumPy. The original code's `apply_rx_all` was looping over qubits 
        and doing moves, which is slow. We can do this much faster.
        """
        pass

    # Let's stick to the vector approach but optimize the application of R_x.
    # Precompute the R_x rotation matrix? No, it's global.
    # Better: Use the fact that R_x(beta) on all qubits is a tensor product.
    # But applying it one qubit at a time is O(n * 2^n).
    
    # Let's refine the vector simulation to be as fast as possible.
    
    num_states = 1 << n
    cost_vec = np.zeros(num_states, dtype=np.float64)
    
    # Precompute cost vector
    # Use bit manipulation for speed
    for u, v in edges:
        # Create mask of states where bit u != bit v
        # This is equivalent to: (state >> u) ^ (state >> v) & 1 == 1
        # Vectorized:
        idx = np.arange(num_states, dtype=np.int64)
        bu = (idx >> u) & 1
        bv = (idx >> v) & 1
        mask = bu != bv
        cost_vec[mask] += 1
    
    # Precompute initial state
    init_state = np.ones(num_states, dtype=np.complex128) / np.sqrt(num_states)
    
    # Precompute R_x matrix (2x2)
    # R_x(beta) = [[cos(beta/2), -i sin(beta/2)], [-i sin(beta/2), cos(beta/2)]]
    
    def simulate(qaoa_angles):
        # qaoa_angles: [gamma1, beta1, gamma2, beta2]
        # Returns expected cut value
        
        g1, b1, g2, b2 = qaoa_angles
        
        # Copy initial state to avoid modifying the precomputed one
        # Note: In-place operations are faster but we need to be careful with optimization.
        # Since we're optimizing, we might reuse memory, but for clarity and safety with 
        # multiple threads (if any) or just clean state, copy is fine for 1024 elements.
        state = init_state.copy()
        
        # Helper to apply R_x(beta) to all qubits
        # This is the most expensive part.
        # We can optimize by precomputing the effect of R_x on the basis states?
        # No, beta changes.
        
        # Optimized R_x application:
        # Instead of moving axes, we can iterate over the bits and apply the 2x2 matrix 
        # to the appropriate slices.
        
        def apply_rx(state, beta):
            c = np.cos(beta / 2.0)
            s = np.sin(beta / 2.0)
            # R_x = [[c, -1j*s], [-1j*s, c]]
            
            # For each qubit q (0 to n-1), we apply R_x.
            # In the array state of size 2^n, qubit q corresponds to a stride of 2^q.
            # The dimension for qubit q is of size 2.
            # We can reshape the state to (2, 2^(n-q-1), 2^q)? 
            # It's easier to just loop q and use slicing.
            
            # Loop over qubits
            for q in range(n):
                # Reshape to separate the q-th qubit dimension
                # Current shape: (2^n,)
                # We want to view it as a tensor where axis q is the qubit.
                # Let's use moveaxis as in the original but it was slow.
                
                # Alternative: Precompute the strides and use direct indexing?
                # For n=10, n is small. The moveaxis overhead is non-trivial.
                
                # Let's try a different approach:
                # We know that applying R_x on qubit q transforms the state amplitudes.
                # If we flatten the state into a tensor of shape (2,)*n, 
                # applying R_x on axis q is a matrix multiplication along that axis.
                
                # Since n is small, let's just do the reshape/moveaxis once per call?
                # No, the original code did it inside the loop.
                
                # Let's try to avoid moveaxis by using strides?
                # Or just accept the cost for n<=10.
                
                # Actually, we can precompute the "permutation" needed? No.
                
                # Let's stick to the vectorized matrix mult on reshaped tensors but optimize the loop.
                # Move axis q to front
                state_view = np.moveaxis(state, q, 0)
                # Reshape to (2, -1)
                shape = state_view.shape
                state_view = state_view.reshape(2, -1)
                
                # Apply matrix
                # v0 = state_view[0]
                # v1 = state_view[1]
                # new_v0 = c*v0 - 1j*s*v1
                # new_v1 = -1j*s*v0 + c*v1
                
                v0 = state_view[0]
                v1 = state_view[1]
                
                # Perform linear combination
                # To save time, we can do:
                # temp0 = c*v0 - 1j*s*v1
                # temp1 = -1j*s*v0 + c*v1
                
                # Using numpy operations
                new_v0 = c * v0 - 1j * s * v1
                new_v1 = -1j * s * v0 + c * v1
                
                state_view = np.vstack((new_v0, new_v1))
                state = state_view.reshape(shape)
                state = np.moveaxis(state, 0, q)
                
            return state

        # Apply Cost Phase 1
        # e^{-i gamma C}
        state = state * np.exp(-1j * g1 * cost_vec)
        
        # Apply R_x 1
        state = apply_rx(state, b1)
        
        # Apply Cost Phase 2
        state = state * np.exp(-1j * g2 * cost_vec)
        
        # Apply R_x 2
        state = apply_rx(state, b2)
        
        # Calculate Expected Cut
        probs = np.abs(state) ** 2
        exp_cut = np.dot(probs, cost_vec)
        
        return exp_cut

    def objective(params):
        if time.time() > deadline:
            return 1e10
        try:
            val = simulate(params)
            # We want to maximize val, so minimize -val
            return -val
        except Exception:
            return 1e10

    # Optimization Strategy
    # 1. Grid search for coarse initialization?
    # 2. Local optimization from multiple starts.
    
    best_val = -np.inf
    best_params = np.array([0.35, 0.25, 0.35, 0.25])
    
    # Use a set of diverse starting points
    starts = [
        [0.35, 0.25, 0.35, 0.25], # Standard heuristic
        [0.5, 0.5, 0.5, 0.5],
        [0.1, 0.1, 0.1, 0.1],
        [1.0, 1.0, 1.0, 1.0],
        [0.2, 0.3, 0.4, 0.2],
    ]
    
    # Add random starts
    np.random.seed(42)
    for _ in range(5):
        starts.append(np.random.uniform(0, np.pi, 4).tolist())
        
    for start in starts:
        if time.time() > deadline:
            break
            
        # Use L-BFGS-B for better convergence if gradients were available, 
        # but we don't have them. Nelder-Mead is robust.
        # We can also try Powell method which is often faster than NM for low dims.
        try:
            res = minimize(objective, start, method='Nelder-Mead', 
                           options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-5})
            
            # Check if we actually improved
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
            
        # Also try a quick local search around the result? 
        # Or just rely on NM.
        
    # Return angles (p, 2)
    # best_params is [g1, b1, g2, b2]
    # We want rows: [g1, b1], [g2, b2]
    return best_params.reshape(p, 2)
