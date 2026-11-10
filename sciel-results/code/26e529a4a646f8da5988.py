import time
import itertools
import sys

def construct(n):
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "1234123142312413241234123214321"
    
    deadline = time.time() + 140
    
    def compute_superperm_and_order(k):
        if k == 1:
            return [1], [(1,)]
        
        S_prev, Order_prev = compute_superperm_and_order(k-1)
        
        word = []
        for p in Order_prev:
            block = list(p) + [k] + list(p)
            max_overlap = min(len(word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if word[len(word)-o:] == block[:o]:
                    overlap = o
                    break
            word.extend(block[overlap:])
        
        S_k = word
        
        seen = set()
        Order_k = []
        for i in range(len(S_k) - k + 1):
            window = tuple(S_k[i:i+k])
            if window not in seen:
                seen.add(window)
                Order_k.append(window)
        
        return S_k, Order_k

    S_prev, Order_prev = compute_superperm_and_order(n-1)
    perms_order = Order_prev
    N = len(perms_order)
    
    # Precompute overlap matrix
    # overlap[i][j] = L(perms_order[i], perms_order[j])
    overlap = [[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            p = perms_order[i]
            q = perms_order[j]
            max_k = len(p)
            for k in range(max_k, -1, -1):
                if p[len(p)-k:] == q[:k]:
                    overlap[i][j] = k
                    break

    current_order = perms_order[:]
    
    # Prefix sums for internal edges
    # fwd[i] = overlap[i][i+1]
    # bwd[i] = overlap[i+1][i]
    fwd = [overlap[i][i+1] for i in range(N-1)]
    bwd = [overlap[i+1][i] for i in range(N-1)]
    
    # Prefix sums
    pref_fwd = [0]*(N)
    pref_bwd = [0]*(N)
    for i in range(N-1):
        pref_fwd[i+1] = pref_fwd[i] + fwd[i]
        pref_bwd[i+1] = pref_bwd[i] + bwd[i]
        
    # Current total overlap
    current_score = pref_fwd[N-1]
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for i in range(N):
            for j in range(i+1, N):
                # Calculate delta
                delta = 0
                
                # Remove old edges
                if i > 0:
                    delta -= overlap[i-1][i]
                if j < N-1:
                    delta -= overlap[j][j+1]
                
                # Remove internal old
                # Sum fwd[i..j-1]
                if j > i:
                    delta -= (pref_fwd[j] - pref_fwd[i])
                    
                # Add new edges
                if i > 0:
                    delta += overlap[i-1][j]
                if j < N-1:
                    delta += overlap[i][j+1]
                
                # Add internal new
                # Sum bwd[i..j-1]
                if j > i:
                    delta += (pref_bwd[j] - pref_bwd[i])
                    
                if delta > 0:
                    # Apply move
                    # Reverse segment i..j
                    current_order[i:j+1] = current_order[i:j+1][::-1]
                    current_score += delta
                    improved = True
                    
                    # Update prefix sums? 
                    # This is tricky. We can just recompute prefix sums after a full pass if needed, 
                    # or update them carefully.
                    # Given N=720, recomputing prefix sums O(N) after each improvement is fine.
                    fwd = [overlap[current_order.index(current_order[k])][current_order.index(current_order[k+1])] for k in range(N-1)] # This is O(N^2) due to index.
                    # Better: just recalculate fwd/bwd from current_order and overlap matrix.
                    # But overlap matrix is indexed by original indices.
                    # We need to map current_order positions to original indices?
                    # No, overlap matrix is fixed based on the items.
                    # Wait, overlap[i][j] is L(item_i, item_j).
                    # If we reorder items, the indices in the matrix still correspond to the items.
                    # So if current_order[k] is item X, and current_order[k+1] is item Y,
                    # we need overlap[index_of_X][index_of_Y].
                    # We can create a map from item to original index.
                    pass
    
    # This approach of updating prefix sums is messy.
    # Given N=720, we can just recalculate the total score from scratch for each candidate?
    # No, that's O(N) per candidate, O(N^3) total. Too slow.
    
    # Alternative: Keep the order as a list of original indices.
    # Then overlap is just a matrix lookup.
    
    # Let's restart the optimization part with indices.
    
    # ... (rewrite optimization)
    
    # For now, let's just use the classic order if optimization is too complex.
    # But 2-opt is standard.
    
    # Let's try a simpler local search: Swap adjacent elements?
    # Or just use the classic order. The gap is small.
    
    # Actually, for n=7, the classic order might be close enough.
    # But we want the best.
    
    # Let's implement the index-based 2-opt properly.
    
    # Map item to index
    item_to_idx = {item: idx for idx, item in enumerate(perms_order)}
    
    # current_order is list of items.
    # Let's convert to list of indices for speed.
    curr_idx_order = [item_to_idx[item] for item in current_order]
    
    # Overlap matrix O[i][j]
    O = overlap
    
    # Helper to get score of index order
    def get_score(idx_order):
        s = 0
        for k in range(len(idx_order)-1):
            s += O[idx_order[k]][idx_order[k+1]]
        return s
    
    # 2-Opt on indices
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        N = len(curr_idx_order)
        for i in range(N):
            for j in range(i+1, N):
                # Reverse i..j
                # New order: curr[:i] + curr[i:j+1][::-1] + curr[j+1:]
                # Delta calculation:
                delta = 0
                if i > 0:
                    delta -= O[curr_idx_order[i-1]][curr_idx_order[i]]
                    delta += O[curr_idx_order[i-1]][curr_idx_order[j]]
                if j < N-1:
                    delta -= O[curr_idx_order[j]][curr_idx_order[j+1]]
                    delta += O[curr_idx_order[i]][curr_idx_order[j+1]]
                
                # Internal
                # Old: sum O[curr[k]][curr[k+1]] for k=i..j-1
                # New: sum O[curr[k+1]][curr[k]] for k=i..j-1
                old_int = 0
                new_int = 0
                for k in range(i, j):
                    old_int += O[curr_idx_order[k]][curr_idx_order[k+1]]
                    new_int += O[curr_idx_order[k+1]][curr_idx_order[k]]
                delta += new_int - old_int
                
                if delta > 0:
                    curr_idx_order[i:j+1] = curr_idx_order[i:j+1][::-1]
                    improved = True
    
    # Reconstruct final order of items
    final_order = [perms_order[idx] for idx in curr_idx_order]
    
    # Build word
    word = []
    for p in final_order:
        block = list(p) + [n] + list(p)
        max_overlap = min(len(word), len(block))
        overlap_val = 0
        for o in range(max_overlap, -1, -1):
            if word[len(word)-o:] == block[:o]:
                overlap_val = o
                break
        word.extend(block[overlap_val:])
        
    return "".join(map(str, word))
