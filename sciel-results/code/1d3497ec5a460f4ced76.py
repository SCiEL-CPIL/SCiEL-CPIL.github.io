import random
import math

def construct():
    # Calculate LCM of 1 to 30
    L = 1
    for i in range(1, 31):
        L = L * i // math.gcd(L, i)
    
    weights = [L // i for i in range(1, 31)]
    N = 30
    
    # --- LLL Basis for Kernel ---
    # Kernel of w: vectors v such that sum(v[i]*w[i]) == 0
    # Basis: v_j = w_j * e_1 - w_1 * e_j for j in 1..29 (0-indexed j=1..29)
    # Note: e_i is unit vector.
    # v_j has w_j at index 0, -w_1 at index j, 0 elsewhere.
    
    basis = []
    w1 = weights[0]
    for j in range(1, N):
        v = [0] * N
        v[0] = weights[j]
        v[j] = -w1
        basis.append(v)
    
    # Simple LLL implementation for integer vectors
    def lll(basis):
        # Basis is a list of lists
        # Gram-Schmidt
        n = len(basis[0])
        b = [v[:] for v in basis]
        b_star = [v[:] for v in basis]
        
        # Initialize b_star
        for i in range(len(b)):
            b_star[i] = b[i][:]
            for j in range(i):
                # mu_{i,j}
                dot_bb = sum(x*y for x,y in zip(b[j], b[j]))
                if dot_bb == 0: continue
                mu = sum(x*y for x,y in zip(b[i], b[j])) / dot_bb
                b_star[i] = [x - mu*y for x,y in zip(b_star[i], b[j])]
        
        # Reduce
        for i in range(len(b)-1, 0, -1):
            for j in range(i-1, -1, -1):
                # Recalculate mu
                dot_bb = sum(x*y for x,y in zip(b[j], b[j]))
                if dot_bb == 0: continue
                mu = sum(x*y for x,y in zip(b[i], b[j])) / dot_bb
                b_star[i] = [x - mu*y for x,y in zip(b_star[i], b[j])]
                
                # Swap if needed
                # Check Lovasz condition
                # |b_star[j]|^2 >= (3/4) |b_star[i-1]|^2 ? 
                # Standard LLL: if |b_star[i]|^2 < (3/4) |b_star[i-1]|^2, swap
                # We need to be careful with the indices in the standard algorithm.
                
                # Simplified LLL step:
                # 1. Update mu and b_star
                # 2. If |b_star[i]|^2 < 0.75 * |b_star[i-1]|^2, swap b[i] and b[i-1] and restart
                # This simplified version is O(n^2) and might be slow for n=29, but acceptable.
                
                # Let's just do a few iterations of swapping based on norms
                pass # Placeholder, will implement proper LLL below

    # Proper LLL
    def lll_reduce(basis):
        n = len(basis[0])
        b = [v[:] for v in basis]
        b_star = [v[:] for v in basis]
        m = len(b)
        
        # Initialize
        for i in range(m):
            b_star[i] = b[i][:]
            for j in range(i):
                dot_bb = sum(x*x for x in b[j])
                if dot_bb == 0: continue
                mu = sum(x*y for x,y in zip(b[i], b[j])) / dot_bb
                b_star[i] = [x - mu*y for x,y in zip(b_star[i], b[j])]
        
        k = 1
        while k < m:
            # Reduce b[k] with respect to b[0..k-1]
            for j in range(k-1, -1, -1):
                dot_bb = sum(x*x for x in b[j])
                if dot_bb == 0: continue
                mu = sum(x*y for x,y in zip(b[k], b[j])) / dot_bb
                b_star[k] = [x - mu*y for x,y in zip(b_star[k], b[j])]
                
                # Integer rounding
                mu_int = int(round(mu))
                if mu_int != 0:
                    b[k] = [x - mu_int*y for x,y in zip(b[k], b[j])]
                    # Update b_star for i > k
                    for i in range(k+1, m):
                        b_star[i] = [x - mu_int*y for x,y in zip(b_star[i], b[j])]
            
            # Check Lovasz condition
            # |b_star[k]|^2 >= 3/4 |b_star[k-1]|^2
            norm_k_sq = sum(x*x for x in b_star[k])
            norm_km1_sq = sum(x*x for x in b_star[k-1]) if k > 0 else 0
            
            if norm_k_sq < 0.75 * norm_km1_sq:
                # Swap
                b[k], b[k-1] = b[k-1], b[k]
                # Recalculate b_star for swapped elements? 
                # Easiest: restart the reduction from k-1
                # But we need to update b_star for all i >= k-1
                # For simplicity, let's just recompute b_star for the whole basis?
                # Or just swap and re-run the inner loop for k-1?
                
                # Recompute b_star for k-1 and k
                # b_star[k-1] is unchanged? No, b[k-1] changed.
                # This simple implementation is tricky.
                # Let's use a more robust simple LLL:
                
                # Swap b[k] and b[k-1]
                b[k], b[k-1] = b[k-1], b[k]
                
                # Recompute b_star for i >= k-1
                for i in range(k-1, m):
                    b_star[i] = b[i][:]
                    for j in range(i):
                        dot_bb = sum(x*x for x in b[j])
                        if dot_bb == 0: continue
                        mu = sum(x*y for x,y in zip(b[i], b[j])) / dot_bb
                        b_star[i] = [x - mu*y for x,y in zip(b_star[i], b[j])]
                
                k = max(0, k-1)
            else:
                k += 1
        
        return b

    kernel_basis = lll_reduce(basis)
    
    best_c = [0] * N
    best_val = float('inf')
    
    # Local Search with Kernel Moves
    for restart in range(200):
        # Random init
        c = [random.choice([-1, 0, 1]) for _ in range(N)]
        current_sum = sum(c[i] * weights[i] for i in range(N))
        
        if current_sum == 0:
             # Force non-zero
             idx = random.randint(0, N-1)
             c[idx] = 1 if c[idx] == -1 else -1
             current_sum = sum(c[i] * weights[i] for i in range(N))
        
        if abs(current_sum) < best_val:
            best_val = abs(current_sum)
            best_c = c[:]
            if best_val == 1:
                return best_c
        
        # Local Search
        for _ in range(500):
            if abs(current_sum) == 1:
                return c
            
            # 1. Try Single Bit Flips
            best_move = None
            best_new_sum = current_sum
            best_new_c = None
            
            for i in range(N):
                old_val = c[i]
                for new_val in [-1, 0, 1]:
                    if new_val == old_val: continue
                    delta = (new_val - old_val) * weights[i]
                    new_sum = current_sum + delta
                    if new_sum == 0: continue
                    if abs(new_sum) < abs(current_sum):
                        # Improvement
                        new_c = c[:]
                        new_c[i] = new_val
                        best_move = new_c
                        best_new_sum = new_sum
                        # Break early? No, find the BEST improvement
                        # But we only care if it's better than current.
                        # Let's just take the first improvement to keep it fast?
                        # Or find the one with smallest abs(new_sum)?
                        # Let's just take the first one to save time, or the best one.
                        # Let's check all and pick the best.
                        # This is O(N) per step.
                        pass
            
            # Re-do the best flip search properly
            min_abs_sum = abs(current_sum)
            best_flip_c = None
            best_flip_sum = current_sum
            for i in range(N):
                old_val = c[i]
                for new_val in [-1, 0, 1]:
                    if new_val == old_val: continue
                    delta = (new_val - old_val) * weights[i]
                    new_sum = current_sum + delta
                    if new_sum == 0: continue
                    if abs(new_sum) < min_abs_sum:
                        min_abs_sum = abs(new_sum)
                        best_flip_c = c[:]
                        best_flip_c[i] = new_val
                        best_flip_sum = new_sum
            
            if best_flip_c is not None:
                c = best_flip_c
                current_sum = best_flip_sum
                continue
            
            # 2. Try Kernel Moves (No change in sum)
            # If no improving flip, try to move in kernel to escape plateau
            moved = False
            for b in kernel_basis:
                # Try adding b
                new_c = [c[i] + b[i] for i in range(N)]
                if all(v in [-1, 0, 1] for v in new_c):
                    # Check if this move leads to a better state?
                    # Sum is same.
                    # Just take it to change configuration.
                    # But we might cycle.
                    # Let's only take it with some probability or if it seems "random"
                    if random.random() < 0.5:
                        c = new_c
                        moved = True
                        break
                # Try subtracting b
                new_c = [c[i] - b[i] for i in range(N)]
                if all(v in [-1, 0, 1] for v in new_c):
                    if random.random() < 0.5:
                        c = new_c
                        moved = True
                        break
            
            if not moved:
                # Stuck
                break

    return best_c
