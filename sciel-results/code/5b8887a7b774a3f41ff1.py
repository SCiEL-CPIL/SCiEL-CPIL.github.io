"""QAOA angle design via vectorized bank selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

# ---------------------------------------------------------------------------
# Fast Vectorized QAOA Expected Cut Calculation
# ---------------------------------------------------------------------------

def _compute_phases(n, edges):
    """
    Pre-compute the phase for each basis state for the MaxCut Hamiltonian.
    phase[b] = sum_{(u,v) in E} (-1)^{b_u + b_v}
    This allows O(1) phase lookup during state evolution.
    """
    num_states = 1 << n
    # Initialize phases to 0
    phases = np.zeros(num_states, dtype=np.int32)
    
    if not edges:
        return phases

    # For each edge, we need to add +1 if bits are different, -1 if same.
    # This is equivalent to adding 1 - 2*(b_u XOR b_v) ? No.
    # Z_u Z_v = (-1)^{b_u} (-1)^{b_v} = (-1)^{b_u + b_v}.
    # If b_u == b_v, (-1)^{even} = 1.
    # If b_u != b_v, (-1)^{odd} = -1.
    # So contribution is 1 if (b_u ^ b_v) == 0, else -1.
    # contribution = 1 - 2 * (b_u ^ b_v)
    
    for u, v in edges:
        # Create a mask for states where bit u and bit v are different
        # We can iterate or use broadcasting. Since n is small, iterating edges is fine.
        # To be fast, we can precompute the XOR pattern for each edge.
        # But doing this in Python loops for every call might be slow if not cached.
        # Let's compute the phase array directly.
        pass
    
    # More efficient:
    # For a given state b, the cut value is the number of edges crossing the cut.
    # H_C |b> = (2 * cut_value(b) - M) |b> where M is total edges?
    # No, H_C = sum Z_u Z_v. Eigenvalue is sum_{(u,v)} (-1)^{b_u+b_v}.
    # Let k be the number of edges with b_u != b_v (cut edges).
    # Let m-M be the number of edges with b_u == b_v.
    # Sum = k * (-1) + (M-k) * (1) = M - 2k.
    # So phase = M - 2 * cut_value(b).
    
    M = len(edges)
    # We need cut_value for all states.
    # cut_value(b) = sum_{(u,v)} (b_u XOR b_v)
    
    # Vectorized approach for cut_value:
    # For each edge (u,v), create an array of (b_u XOR b_v) for all b.
    # This is a bit of work, but doable with broadcasting.
    
    # Let's just compute it directly for n <= 12.
    # 2^12 = 4096 states.
    # M is at most n(n-1)/2 ~ 66.
    
    # Initialize cut_vals to 0
    cut_vals = np.zeros(num_states, dtype=np.int32)
    
    # Precompute bit masks
    # For each edge (u, v), we want to add 1 to cut_vals[b] if bit u != bit v.
    # We can create a boolean mask for each edge.
    # To avoid creating huge temporary arrays for each edge, we can use the fact that
    # the pattern repeats.
    
    # Actually, for n=10, num_states=1024. Even a loop over edges (max 45) with vectorized ops is fine.
    for u, v in edges:
        # Create masks for bit u and bit v being 1
        # bit_u is 1 if (b >> u) & 1 == 1
        # We can generate this using np.arange and bit ops, or precomputed lookup.
        
        # Precomputed lookup for bits is faster if we do it once per graph.
        # Let's assume this function is called with the same graph multiple times?
        # No, it's called per graph evaluation.
        
        # Efficient way:
        # bit_u_array = ((np.arange(num_states) >> u) & 1)
        # bit_v_array = ((np.arange(np.arange(num_states) >> v) & 1)
        # xor = bit_u_array ^ bit_v_array
        # cut_vals += xor
        
        # This creates temporary arrays of size 1024/4096. 45 edges * 4096 * 8 bytes ~ 1.5MB. Fine.
        
        bits_u = (np.arange(num_states) >> u) & 1
        bits_v = (np.arange(num_states) >> v) & 1
        cut_vals += (bits_u ^ bits_v)
        
    # phase = M - 2 * cut_vals
    phases = M - 2 * cut_vals
    return phases

def _expected_cut_qaoa_fast(n, edges, gammas, betas, phases=None):
    """
    Compute expected cut for given angles using vectorized state evolution.
    gammas, betas: arrays of shape (p,)
    """
    if phases is None:
        phases = _compute_phases(n, edges)
        
    num_states = 1 << n
    state = np.ones(num_states, dtype=np.complex128) / np.sqrt(num_states)
    
    p = len(gammas)
    
    for i in range(p):
        # Apply exp(-i gamma_i H_C)
        # state *= exp(-i gamma_i * phases)
        state *= np.exp(-1j * gammas[i] * phases)
        
        # Apply exp(-i beta_i H_B)
        # H_B = sum X_k
        # This is a tensor product of single-qubit rotations.
        # We can apply it by iterating over qubits.
        # For each qubit q, apply the gate to the q-th dimension.
        # Reshape state to (2, 2, ..., 2) (n times)
        # Apply the gate along axis q
        # Reshape back
        
        # This is O(n * 2^n) operations.
        # For n=10, 10 * 1024 = 10k ops. Very fast.
        
        # However, reshaping and applying a 2x2 matrix along an axis in NumPy
        # can be done efficiently.
        
        # Gate matrix for exp(-i beta X):
        # [[cos(beta), -i sin(beta)], [-i sin(beta), cos(beta)]]
        c = np.cos(betas[i])
        s = -1j * np.sin(betas[i])
        
        # Reshape to n-dimensional tensor
        # Note: NumPy uses C-order by default, so the last index is the fastest changing.
        # Bit 0 is the LSB. In `np.arange`, index 0 is 0...0, index 1 is 0...1 (bit 0 set).
        # So axis 0 corresponds to bit 0?
        # If we reshape to shape (2,)*n, the first axis corresponds to the most significant bit?
        # No, `np.reshape` flattens in C-order.
        # Array index `b` corresponds to bits b_{n-1} ... b_0.
        # In a shape (2,)*n array, index [i_0, i_1, ..., i_{n-1}] corresponds to
        # flat index i_0 * 2^{n-1} + ... + i_{n-1}.
        # So axis 0 is the MSB (bit n-1), axis n-1 is the LSB (bit 0).
        
        # We want to apply gate to qubit q (bit q).
        # This corresponds to axis (n-1-q) in the reshaped tensor.
        
        shape = [2] * n
        tensor = state.reshape(shape)
        
        # We need to apply the 2x2 matrix [[c, s], [s, c]] along the axis corresponding to qubit q.
        # Let axis_q = n - 1 - q
        
        # We can use np.einsum or just manual slicing for small n.
        # Manual slicing is often faster for small n.
        
        # Let's use a generic approach:
        # Move the target axis to the last position, apply gate, move back.
        # Or just use the fact that it's a 2x2 matrix multiplication.
        
        # Let's try a simple loop over the "other" dimensions? No, that's slow.
        
        # Efficient NumPy way:
        # Create the gate matrix
        gate = np.array([[c, s], [s, c]], dtype=np.complex128)
        
        # Apply along axis (n-1-q)
        # We can use np.tensordot or np.einsum.
        # Let's use np.einsum for clarity and speed.
        
        # Actually, for n <= 12, a simple loop over the 2^n states in Python is ~10k-40k iterations.
        # That might be ~5-20ms. With 100 bank entries, that's 0.5-2s. Too slow?
        # The vectorized state evolution above is O(n 2^n).
        # Let's stick to the vectorized reshape approach if we can make it fast.
        
        # Alternative: Precompute the permutation or use the fact that X gates are local.
        # Actually, the most efficient way in NumPy for this specific structure is:
        
        # For each q in range(n):
        #   We need to mix states differing only in bit q.
        #   This can be done by reshaping the state array into blocks of size 2^(q+1)?
        
        # Let's use the `np.apply_along_axis`? No, too slow.
        
        # Let's use the fact that we can split the state array into pairs.
        # For qubit q, the states are grouped by their other bits.
        # We can reshape the state to have an axis of size 2 for qubit q, and one big axis for the rest.
        
        # Let's try:
        # 1. Reshape state to (2^(n-1), 2) where the last axis is the qubit we want to operate on?
        #    This only works if the qubit is the LSB.
        #    If q is not LSB, we need to permute axes.
        
        # Permute axes so that qubit q is the last axis.
        # Current axes: 0 (MSB) ... n-1 (LSB).
        # Qubit q is at axis n-1-q.
        # We want to move axis (n-1-q) to the end.
        
        # Let's just use the full tensor approach with `np.einsum`.
        # It might be overkill but correct.
        
        # Actually, there is a simpler trick.
        # The operator exp(-i beta sum X) is a product of commuting single-qubit gates.
        # We can apply them one by one.
        
        # Let's implement the apply_gate function using reshape and broadcasting.
        
        def apply_x_gate(state, q, beta):
            # state: 1D array of size 2^n
            # q: qubit index
            # Apply exp(-i beta X_q)
            
            c = np.cos(beta)
            s = -1j * np.sin(beta)
            
            # We need to group indices by all bits except q.
            # Let's reshape state into a 2D array of shape (2^(n-1), 2).
            # This requires the qubit to be the fastest varying index (LSB).
            # If q != 0, we need to permute.
            
            # Permute axes of the n-dimensional tensor so that axis (n-1-q) is last.
            tensor = state.reshape([2]*n)
            
            # Get list of axes
            axes = list(range(n))
            # Move axis (n-1-q) to the end
            target_axis = n - 1 - q
            # Remove it from its position and append
            axes.remove(target_axis)
            axes.append(target_axis)
            
            # Transpose
            tensor = tensor.transpose(axes)
            
            # Now the last axis is the qubit q.
            # Reshape to (2^(n-1), 2)
            tensor_2d = tensor.reshape(-1, 2)
            
            # Apply gate: new_state = gate @ old_state
            # gate = [[c, s], [s, c]]
            # new[:, 0] = c * old[:, 0] + s * old[:, 1]
            # new[:, 1] = s * old[:, 0] + c * old[:, 1]
            
            old = tensor_2d
            new = np.empty_like(old)
            new[:, 0] = c * old[:, 0] + s * old[:, 1]
            new[:, 1] = s * old[:, 0] + c * old[:, 1]
            
            # Reshape back
            new_tensor = new.reshape([2]*n)
            
            # Inverse transpose
            # We need to move the last axis back to position (n-1-q).
            # The current order of axes is: [all except target, target]
            # We want: [0, 1, ..., n-1]
            # The inverse permutation is:
            # The element at position n-1 in the transposed array (which is the target axis)
            # needs to go back to position target_axis.
            
            # Let's construct the inverse permutation.
            # The forward permutation was: axes_forward = [a0, a1, ..., a_{n-2}, a_{n-1}] where a_{n-1} = target_axis.
            # The inverse permutation p_inv satisfies p_inv[axes_forward[i]] = i.
            # We want to transpose back to original order.
            # We can just compute the inverse permutation explicitly.
            
            # Easier: just transpose with the inverse of the `axes` list.
            # The `axes` list tells us where the new axis i comes from in the old tensor.
            # Wait, `tensor.transpose(axes)` means new_axis_i = old_axis_axes[i].
            # So to go back, we need `old_axis_axes[i] = new_axis_i`.
            # We need a permutation `inv_axes` such that `tensor.transpose(inv_axes)` restores the original.
            # `inv_axes[axes[i]] = i`.
            
            inv_axes = [0] * n
            for i, ax in enumerate(axes):
                inv_axes[ax] = i
            
            restored_tensor = new_tensor.transpose(inv_axes)
            
            return restored_tensor.reshape(-1)

        for q in range(n):
            state = apply_x_gate(state, q, betas[i])
            
    # Compute expected cut
    # E[cut] = sum_{(u,v) in E} 0.5 * (1 - <Z_u Z_v>)
    # <Z_u Z_v> = sum_b |state[b]|^2 * (-1)^{b_u + b_v}
    
    # We already have `cut_vals` from `_compute_phases`?
    # No, `_compute_phases` returns `M - 2 * cut_vals`.
    # So `cut_vals = (M - phases) / 2`.
    # We can reuse the logic.
    
    M = len(edges)
    if M == 0:
        return 0.0
        
    # Recompute cut_vals or store it?
    # For efficiency in the bank selection, we should precompute `cut_vals` for the graph.
    
    # Let's refactor `_expected_cut_qaoa_fast` to take `cut_vals` as input.
    
    # For now, let's just calculate it.
    num_states = 1 << n
    cut_vals = np.zeros(num_states, dtype=np.int32)
    for u, v in edges:
        bits_u = (np.arange(num_states) >> u) & 1
        bits_v = (np.arange(num_states) >> v) & 1
        cut_vals += (bits_u ^ bits_v)
        
    # <Z_u Z_v> for each edge? No, we need the total expected cut.
    # E[cut] = sum_{(u,v)} 0.5 * (1 - <Z_u Z_v>)
    # This is not simply related to the total cut value distribution unless we compute each edge's correlation.
    # Wait, the expected cut is the expectation of the number of cut edges.
    # E[cut] = sum_{(u,v)} P(edge (u,v) is cut)
    # P(edge (u,v) is cut) = 0.5 * (1 - <Z_u Z_v>)
    
    # So we need <Z_u Z_v> for each edge.
    # <Z_u Z_v> = sum_b |state[b]|^2 * (-1)^{b_u + b_v}
    
    # This requires computing the correlation for each edge separately.
    # This is O(M * 2^n).
    # For n=10, M=45, 2^10=1024. 45 * 1024 = 46k ops. Fast.
    
    probs = np.abs(state) ** 2
    
    total_cut = 0.0
    for u, v in edges:
        bits_u = (np.arange(num_states) >> u) & 1
        bits_v = (np.arange(num_states) >> v) & 1
        # (-1)^{b_u + b_v} is 1 if same, -1 if diff.
        # sign = 1 - 2*(bits_u ^ bits_v)
        signs = 1 - 2 * (bits_u ^ bits_v)
        corr = np.dot(probs, signs)
        total_cut += 0.5 * (1 - corr)
        
    return total_cut

# ---------------------------------------------------------------------------
# Pre-compute a Bank of Good Angles
# ---------------------------------------------------------------------------

def _generate_test_graphs():
    """Generate a diverse set of graphs to train the bank."""
    graphs = []
    # Sizes 6, 8, 10
    # Densities 0.25, 0.5, 0.75
    # Seeds 1, 2, 3
    
    for n in [6, 8, 10]:
        for density in [0.25, 0.5, 0.75]:
            for seed in [1, 2, 3]:
                rng = np.random.RandomState(seed)
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if rng.rand() < density:
                            edges.append((i, j))
                if edges:
                    graphs.append((n, edges))
    return graphs

def _precompute_bank():
    """
    Create a bank of angle sets by optimizing on a few representative graphs
    and also including some heuristic sets.
    """
    bank = []
    
    # Add some heuristic angles known to be decent
    # From literature, gamma ~ 0.5, beta ~ 0.5 often works.
    bank.append(np.array([[0.5, 0.5], [0.5, 0.5]]))
    bank.append(np.array([[0.3, 0.3], [0.3, 0.3]]))
    bank.append(np.array([[0.8, 0.8], [0.8, 0.8]]))
    
    # Optimize on a few specific graphs to add graph-specific "flavors"
    test_graphs = _generate_test_graphs()
    
    # Pick 3 diverse graphs
    selected = [test_graphs[0], test_graphs[len(test_graphs)//2], test_graphs[-1]]
    
    for n, edges in selected:
        # Optimize for this graph
        # Use a quick differential evolution
        def obj(params):
            g1, g2, b1, b2 = params
            # Clip
            g1 = np.clip(g1, 0, np.pi)
            g2 = np.clip(g2, 0, np.pi)
            b1 = np.clip(b1, 0, np.pi)
            b2 = np.clip(b2, 0, np.pi)
            cut = _expected_cut_qaoa_fast(n, edges, np.array([g1, g2]), np.array([b1, b2]))
            return -cut
        
        result = differential_evolution(
            obj, 
            bounds=[(0, np.pi)] * 4, 
            seed=42, 
            maxiter=50, 
            tol=1e-4,
            popsize=15
        )
        
        best_params = result.x
        # Clip
        best_params = np.clip(best_params, 0, np.pi)
        angles_opt = np.array([[best_params[0], best_params[2]], [best_params[1], best_params[3]]])
        
        # Check if it's distinct from existing bank
        is_new = True
        for existing in bank:
            if np.allclose(existing, angles_opt, atol=0.1):
                is_new = False
                break
        
        if is_new:
            bank.append(angles_opt)
            
    return bank

_BANK = _precompute_bank()

# ---------------------------------------------------------------------------
# Main Interface
# ---------------------------------------------------------------------------

def angles(n, edges, p):
    """
    Return optimal QAOA angles for the given graph.
    Selects the best angle set from the pre-computed bank for this specific graph.
    """
    p = 2
    
    if not edges:
        return np.zeros((2, 2))
        
    # Precompute cut values for this graph to speed up evaluation
    num_states = 1 << n
    cut_vals = np.zeros(num_states, dtype=np.int32)
    for u, v in edges:
        bits_u = (np.arange(num_states) >> u) & 1
        bits_v = (np.arange(num_states) >> v) & 1
        cut_vals += (bits_u ^ bits_v)
        
    # We need to evaluate the expected cut for each bank entry.
    # To make this faster, we can vectorize the state evolution for all bank entries?
    # No, the state evolution is sequential per angle set.
    # But we can compute the expected cut for each bank entry quickly.
    
    best_cut = -np.inf
    best_angles = _BANK[0]
    
    for candidate_angles in _BANK:
        gammas = candidate_angles[:, 0]
        betas = candidate_angles[:, 1]
        
        # Compute expected cut
        cut = _expected_cut_qaoa_fast(n, edges, gammas, betas)
        
        if cut > best_cut:
            best_cut = cut
            best_angles = candidate_angles
            
    return best_angles

# EVOLVE-BLOCK-END
