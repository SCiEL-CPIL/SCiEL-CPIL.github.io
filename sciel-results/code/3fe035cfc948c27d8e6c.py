import math
import time
from functools import reduce
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin

    n = 30
    # Calculate LCM of 1..30
    H = reduce(math.lcm, range(1, n + 1))
    
    # Weights W_i = H / i
    W = [H // i for i in range(1, n + 1)]
    
    # Construct the lattice basis
    # Dimension D = n + 1
    D = n + 1
    # Basis B is a list of D vectors, each of length D
    # B[i] = e_i + W[i] * e_{n} for i in 0..n-1
    # B[n] = e_{n}
    
    B = []
    for i in range(n):
        vec = [0] * D
        vec[i] = 1
        vec[n] = W[i]
        B.append(vec)
    
    vec_last = [0] * D
    vec_last[n] = 1
    B.append(vec_last)
    
    # Perform LLL reduction
    # We use integer arithmetic. 
    # LLL parameters: delta = 3/4, epsilon = 0
    delta = 3.0 / 4.0
    
    # LLL implementation
    def lll_reduce(B, delta=0.75):
        # B is list of vectors
        # Make a copy to avoid modifying original
        B = [v[:] for v in B]
        D = len(B)
        
        # Initialize mu and lambda
        # mu[i][j] = <b_i, b_j> / <b_j, b_j>
        # lambda[i][j] = <b_i, b_j> - mu[i][j] * <b_j, b_j>
        
        # We need Gram-Schmidt orthogonalization
        # b_i^* = b_i - sum_{j<i} mu[i][j] b_j^*
        
        # For integer arithmetic, we can use rational numbers or just floats 
        # since D is small (31) and numbers are not astronomically large 
        # (H is around 10^12, W is around 10^11).
        # Floats should be precise enough for LLL in this dimension.
        
        # Let's use floats for mu and lambda, but keep B as integers.
        
        # Initialize
        mu = [[0.0] * D for _ in range(D)]
        lam = [[0.0] * D for _ in range(D)]
        
        # Gram-Schmidt
        # b_0^* = b_0
        # mu[0][0] = 1
        # lam[0][0] = <b_0, b_0>
        
        for i in range(D):
            for j in range(i + 1):
                if j == i:
                    # <b_i, b_i>
                    dot = sum(b * b for b in B[i])
                    lam[i][i] = float(dot)
                    mu[i][i] = 1.0
                else:
                    # <b_i, b_j>
                    dot = sum(B[i][k] * B[j][k] for k in range(D))
                    # mu[i][j] = <b_i, b_j> / <b_j, b_j>
                    # Note: <b_j, b_j> in GS is lam[j][j]
                    mu[i][j] = dot / lam[j][j]
                    # lam[i][j] = <b_i, b_j> - mu[i][j] * lam[j][j]
                    lam[i][j] = dot - mu[i][j] * lam[j][j]
                    
        # LLL Main Loop
        k = 1
        while k < D:
            # Step 1: Size reduction
            for j in range(k - 1, -1, -1):
                # b_k = b_k - round(mu[k][j]) b_j
                m = round(mu[k][j])
                if m != 0:
                    for l in range(D):
                        B[k][l] -= m * B[j][l]
                    # Update mu and lam
                    for i in range(k + 1):
                        if i < k:
                            # mu[i][k] is not directly updated, but 
                            # we need to update mu[i][j] for j < k? 
                            # No, standard update:
                            # mu[i][k] changes if we modify b_k? No, b_k is modified.
                            # mu[i][j] for j < k are unaffected by b_k modification?
                            # Actually, we need to recompute GS for the modified b_k.
                            # It's easier to recompute mu and lam for row k and col k.
                            pass
                            
                    # Recompute mu[k][j] and lam[k][j]
                    # And update mu[i][k] and lam[i][k] for i > k?
                    # Let's just recompute the entire GS for safety, 
                    # or use the standard update formulas.
                    
                    # Standard Update:
                    # mu[i][k] for i > k:
                    # b_i is unchanged, b_k changed.
                    # mu[i][k] = <b_i, b_k> / <b_k, b_k>
                    # This is getting complex. 
                    # Given D=31, recomputing GS is O(D^3) = 30k ops.
                    # LLL iterations are limited.
                    # Let's recompute GS for the whole basis.
                    for i in range(D):
                        for j in range(i + 1):
                            if j == i:
                                dot = sum(b * b for b in B[i])
                                lam[i][i] = float(dot)
                                mu[i][i] = 1.0
                            else:
                                dot = sum(B[i][l] * B[j][l] for l in range(D))
                                mu[i][j] = dot / lam[j][j]
                                lam[i][j] = dot - mu[i][j] * lam[j][j]
                    
            # Step 2: Lovasz Condition
            if lam[k][k - 1] >= (delta - mu[k][k - 1] ** 2) * lam[k - 1][k - 1]:
                k += 1
            else:
                # Swap b_k and b_{k-1}
                B[k], B[k - 1] = B[k - 1], B[k]
                # Recompute GS
                for i in range(D):
                    for j in range(i + 1):
                        if j == i:
                            dot = sum(b * b for b in B[i])
                            lam[i][i] = float(dot)
                            mu[i][i] = 1.0
                        else:
                            dot = sum(B[i][l] * B[j][l] for l in range(D))
                            mu[i][j] = dot / lam[j][j]
                            lam[i][j] = dot - mu[i][j] * lam[j][j]
                k = max(1, k - 1)
                
        return B

    B_reduced = lll_reduce(B)
    
    # Now, find the shortest non-zero valid vector in this lattice.
    # We use Dijkstra's algorithm on the lattice graph.
    # Nodes are integer combinations of B_reduced.
    # Edge weights are Euclidean norms.
    # We want to find a vector v such that:
    # 1. v is not the zero vector.
    # 2. The first n components of v are in {-1, 0, 1}.
    # 3. The last component (sum) is non-zero.
    
    # Since the lattice is generated by B_reduced, any vector v = sum k_i B_reduced[i].
    # We represent a node by the tuple of coefficients k = (k_0, ..., k_{D-1}).
    
    # Start with k = (0, ..., 0), v = (0, ..., 0), norm = 0.
    # We push neighbors k +/- e_i.
    
    # To avoid infinite loops, we keep a dictionary of visited nodes.
    # However, the number of possible k is infinite.
    # But we stop when the norm exceeds the best valid norm found.
    
    best_val = float('inf')
    best_coeffs = None
    
    # Priority queue: (norm, k_tuple, v_tuple)
    # We store v to avoid recomputing sum.
    
    # Initial state
    k_start = tuple([0] * D)
    v_start = tuple([0] * D)
    norm_start = 0.0
    
    heap = [(norm_start, k_start, v_start)]
    visited = {k_start: norm_start}
    
    # We limit the number of nodes explored to ensure time limit.
    # 100,000 nodes should be fast enough.
    max_nodes = 100000
    nodes_explored = 0
    
    while heap and nodes_explored < max_nodes:
        if time.time() > deadline - 5:
            break
            
        norm, k, v = heapq.heappop(heap)
        
        # Check if this node is the best valid solution
        # v is a tuple of D integers.
        # First n components are coefficients.
        # Last component is the scaled sum.
        
        if v != v_start: # Not zero vector
            coeffs = v[:n]
            sum_val = v[n]
            
            # Check constraints
            valid = True
            for c in coeffs:
                if c < -1 or c > 1:
                    valid = False
                    break
            
            if valid and sum_val != 0:
                abs_sum = abs(sum_val)
                if abs_sum < best_val:
                    best_val = abs_sum
                    best_coeffs = list(coeffs)
                    
                    # If we find a very small sum, we can tighten the bound
                    # But we can't prune the heap easily.
                    # We just continue to see if we can find smaller.
                    
        # Generate neighbors
        for i in range(D):
            # k + e_i
            new_k_list = list(k)
            new_k_list[i] += 1
            new_k = tuple(new_k_list)
            
            if new_k not in visited:
                # new_v = v + B_reduced[i]
                new_v_list = list(v)
                for j in range(D):
                    new_v_list[j] += B_reduced[i][j]
                new_v = tuple(new_v_list)
                
                # Calculate norm
                # norm^2 = sum x^2
                # We can compute incrementally:
                # new_norm^2 = norm^2 + 2 * dot(v, B_reduced[i]) + norm(B_reduced[i])^2
                # To avoid sqrt, we can compare squared norms.
                # But heap needs comparable keys.
                # Let's compute squared norm.
                
                # Precompute norms of basis vectors?
                # No, just compute.
                new_norm_sq = sum(x * x for x in new_v)
                # We store sqrt in heap? Or just squared?
                # If we store squared, we don't need sqrt.
                # Let's store squared norm to save time.
                
                # Update visited with squared norm
                visited[new_k] = new_norm_sq
                heapq.heappush(heap, (new_norm_sq, new_k, new_v))
                
                nodes_explored += 1
                
            # k - e_i
            new_k_list = list(k)
            new_k_list[i] -= 1
            new_k = tuple(new_k_list)
            
            if new_k not in visited:
                new_v_list = list(v)
                for j in range(D):
                    new_v_list[j] -= B_reduced[i][j]
                new_v = tuple(new_v_list)
                
                new_norm_sq = sum(x * x for x in new_v)
                
                visited[new_k] = new_norm_sq
                heapq.heappush(heap, (new_norm_sq, new_k, new_v))
                
                nodes_explored += 1
                
    if best_coeffs is not None:
        return best_coeffs
    
    # Fallback: Greedy
    coeffs = [0] * n
    curr_sum = 0
    for i in range(n, 0, -1):
        w = W[i - 1]
        best_c = 0
        best_val_local = curr_sum
        for c in [-1, 0, 1]:
            val = curr_sum + c * w
            if abs(val) < abs(best_val_local):
                best_val_local = val
                best_c = c
        coeffs[i - 1] = best_c
        curr_sum = best_val_local
        
    if curr_sum == 0:
        min_idx = min(range(n), key=lambda i: W[i])
        if coeffs[min_idx] == 1:
            coeffs[min_idx] = -1
        else:
            coeffs[min_idx] = 1
            
    return coeffs

