import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    def verify(colors, n):
        """
        Returns True if the coloring is valid (no red K4, no blue K5).
        colors is a list of 0s and 1s, one for each triple in lexicographic order.
        """
        if len(colors) != len(list(combinations(range(n), 3))):
            return False
        
        # Build index map for O(1) lookup
        triples = list(combinations(range(n), 3))
        tri_index = {t: i for i, t in enumerate(triples)}
        
        # Check Red K4 (all 4 triples red=0)
        for quad in combinations(range(n), 4):
            a,b,c,d = quad
            i1 = tri_index[(a,b,c)]
            i2 = tri_index[(a,b,d)]
            i3 = tri_index[(a,c,d)]
            i4 = tri_index[(b,c,d)]
            if colors[i1] == 0 and colors[i2] == 0 and colors[i3] == 0 and colors[i4] == 0:
                return False
                
        # Check Blue K5 (all 10 triples blue=1)
        for quint in combinations(range(n), 5):
            verts = list(quint)
            all_blue = True
            for comb in combinations(verts, 3):
                idx = tri_index[comb]
                if colors[idx] != 1:
                    all_blue = False
                    break
            if all_blue:
                return False
        return True

    def build_structure(n):
        triples = list(combinations(range(n), 3))
        tri_index = {t: i for i, t in enumerate(triples)}
        num_triples = len(triples)
        
        # Precompute the indices of triples for each K4 (4-set) and K5 (5-set)
        # This allows fast violation counting.
        
        sets_4 = list(combinations(range(n), 4))
        set4_triples = []
        for s in sets_4:
            a, b, c, d = s
            ts = [
                tri_index[(a, b, c)],
                tri_index[(a, b, d)],
                tri_index[(a, c, d)],
                tri_index[(b, c, d)]
            ]
            set4_triples.append(np.array(ts, dtype=np.int32))
        num_set4 = len(sets_4)
        
        sets_5 = list(combinations(range(n), 5))
        set5_triples = []
        for s in sets_5:
            ts = []
            for i in range(5):
                for j in range(i+1, 5):
                    for k in range(j+1, 5):
                        ts.append(tri_index[(s[i], s[j], s[k])])
            set5_triples.append(np.array(ts, dtype=np.int32))
        num_set5 = len(sets_5)
        
        # Inverse mapping: for each triple, which K4s and K5s does it belong to?
        triple_to_set4 = [[] for _ in range(num_triples)]
        for i, ts in enumerate(set4_triples):
            for t_idx in ts:
                triple_to_set4[t_idx].append(i)
                
        triple_to_set5 = [[] for _ in range(num_triples)]
        for i, ts in enumerate(set5_triples):
            for t_idx in ts:
                triple_to_set5[t_idx].append(i)
                
        return {
            'triples': triples,
            'set4_triples': set4_triples,
            'set5_triples': set5_triples,
            'triple_to_set4': triple_to_set4,
            'triple_to_set5': triple_to_set5,
            'num_triples': num_triples,
            'num_set4': num_set4,
            'num_set5': num_set5
        }

    def init_violations(colors, struct):
        # viol4[i] = number of red triples in K4 i. Violation if viol4[i] == 4.
        # viol5[i] = number of blue triples in K5 i. Violation if viol5[i] == 10.
        num_set4 = struct['num_set4']
        num_set5 = struct['num_set5']
        viol4 = np.zeros(num_set4, dtype=np.int32)
        for i, ts in enumerate(struct['set4_triples']):
            viol4[i] = 4 - np.sum(colors[ts])
            
        viol5 = np.zeros(num_set5, dtype=np.int32)
        for i, ts in enumerate(struct['set5_triples']):
            viol5[i] = np.sum(colors[ts])
            
        total_v = int(np.sum(viol4 == 4) + np.sum(viol5 == 10))
        return viol4, viol5, total_v

    def local_search(n, time_budget, seed):
        random.seed(seed)
        np.random.seed(seed)
        
        struct = build_structure(n)
        num_triples = struct['num_triples']
        
        # Start with a random coloring
        colors = np.random.randint(0, 2, size=num_triples)
        viol4, viol5, total_v = init_violations(colors, struct)
        
        best_colors = colors.copy()
        best_v = total_v
        
        start_ls = time.time()
        
        # Pre-convert lists to arrays for faster indexing if needed, but lists are fine for small counts.
        # We will use the structure to update viol4 and viol5 incrementally.
        
        while time.time() - start_ls < time_budget:
            if best_v == 0:
                break
                
            # Find a triple to flip.
            # Strategy: Pick a random triple involved in a violation, or just a random triple.
            # To be efficient, we can sample a few candidates.
            
            # Let's try to find a move that reduces violations.
            # We sample `sample_size` random triples and pick the one with the best delta.
            sample_size = 50
            best_delta = 0
            best_idx = -1
            
            for _ in range(sample_size):
                t_idx = np.random.randint(0, num_triples)
                old_color = colors[t_idx]
                
                d_v = 0
                s4_indices = struct['triple_to_set4'][t_idx]
                if s4_indices:
                    s4_arr = np.array(s4_indices, dtype=np.int32)
                    old_counts = viol4[s4_arr]
                    if old_color == 0:
                        new_counts = old_counts - 1
                    else:
                        new_counts = old_counts + 1
                    d_v += np.sum((new_counts == 4).astype(np.int32) - (old_counts == 4).astype(np.int32))
                    
                s5_indices = struct['triple_to_set5'][t_idx]
                if s5_indices:
                    s5_arr = np.array(s5_indices, dtype=np.int32)
                    old_counts = viol5[s5_arr]
                    if old_color == 1:
                        new_counts = old_counts - 1
                    else:
                        new_counts = old_counts + 1
                    d_v += np.sum((new_counts == 10).astype(np.int32) - (old_counts == 10).astype(np.int32))
                    
                if d_v < best_delta:
                    best_delta = d_v
                    best_idx = t_idx
            
            if best_idx != -1 and best_delta < 0:
                # Apply the best move found in the sample
                t_idx = best_idx
                old_color = colors[t_idx]
                new_color = 1 - old_color
                
                # Update structures
                s4_indices = struct['triple_to_set4'][t_idx]
                if s4_indices:
                    s4_arr = np.array(s4_indices, dtype=np.int32)
                    if old_color == 0:
                        viol4[s4_arr] -= 1
                    else:
                        viol4[s4_arr] += 1
                        
                s5_indices = struct['triple_to_set5'][t_idx]
                if s5_indices:
                    s5_arr = np.array(s5_indices, dtype=np.int32)
                    if old_color == 1:
                        viol5[s5_arr] -= 1
                    else:
                        viol5[s5_arr] += 1
                        
                colors[t_idx] = new_color
                total_v += best_delta
                
                if total_v < best_v:
                    best_v = total_v
                    best_colors = colors.copy()
            else:
                # No improvement found in sample. 
                # Perform a random flip to escape local minimum.
                # This is a simple random walk step.
                t_idx = np.random.randint(0, num_triples)
                old_color = colors[t_idx]
                new_color = 1 - old_color
                
                s4_indices = struct['triple_to_set4'][t_idx]
                if s4_indices:
                    s4_arr = np.array(s4_indices, dtype=np.int32)
                    if old_color == 0:
                        viol4[s4_arr] -= 1
                    else:
                        viol4[s4_arr] += 1
                            
                s5_indices = struct['triple_to_set5'][t_idx]
                if s5_indices:
                    s5_arr = np.array(s5_indices, dtype=np.int32)
                    if old_color == 1:
                        viol5[s5_arr] -= 1
                    else:
                        viol5[s5_arr] += 1
                        
                colors[t_idx] = new_color
                
                # Recalculate total_v for the random flip? 
                # It's safer to just update it based on the change, but we didn't calculate delta for the random flip.
                # Let's just recalculate total_v from viol arrays if we do random flips? 
                # No, that's expensive.
                # Let's calculate delta for the random flip too, or just accept that total_v might be slightly off?
                # No, total_v must be accurate for the "best_v" check.
                
                # Let's re-calculate delta for the random flip properly.
                # Actually, I can just re-run the delta calc for the random flip.
                # But I already flipped it. 
                # Let's revert and do it properly? 
                # Or just calculate the new total_v from the arrays? 
                # np.sum(viol4 == 4) + np.sum(viol5 == 10) is O(num_sets). 
                # For n=34, num_sets ~ 300k. This is fast enough to do occasionally? 
                # Maybe every 100 steps?
                
                # For now, let's just assume the random flip might increase or decrease v.
                # We can just set total_v = -1 to force a recalculation?
                # Or better: calculate delta for the random flip before applying.
                
                # Let's refactor the flip application to a helper function.
                
                # Actually, let's just recalculate total_v from the arrays to be safe, but only if we did a random flip.
                # This is O(num_set4 + num_set5) which is ~ 300k operations. 
                # If we do this every time we fail to find an improvement, it might be slow if we get stuck.
                # But getting stuck means we are doing random flips, which might be many.
                # Let's limit the random flips or just accept the cost.
                
                # Let's try to calculate the delta for the random flip as well, just like we did for the sampled ones.
                # I can do this by storing the old state? No, I already updated.
                # I can just calculate the delta now by looking at the current state and the fact that I flipped t_idx.
                # But I need the OLD counts to calculate the delta. I don't have them anymore.
                
                # Okay, let's just recalculate total_v from the arrays.
                # It's a numpy sum over ~300k elements, which is very fast (<1ms).
                total_v = int(np.sum(viol4 == 4) + np.sum(viol5 == 10))
                
                if total_v < best_v:
                    best_v = total_v
                    best_colors = colors.copy()

        return best_colors, best_v

    def try_n(n, time_budget, seed_offset=0):
        best_v = float('inf')
        best_colors = None
        
        # Run multiple restarts
        start = time.time()
        attempts = 0
        while time.time() - start < time_budget:
            seed = seed_offset + attempts
            # Give each restart a small time slice
            # If n is large, local search is expensive.
            # Let's give 1 second per restart?
            # Or just run until time_budget is up?
            # local_search has its own time_budget.
            # Let's pass the remaining time to local_search.
            remaining = time_budget - (time.time() - start)
            if remaining < 0.5:
                break
            
            colors, v = local_search(n, min(2.0, remaining), seed)
            if v < best_v:
                best_v = v
                best_colors = colors
            if v == 0:
                break
            attempts += 1
            
        if best_colors is not None and best_v == 0:
            # Verify to be safe
            if verify(best_colors.tolist(), n):
                return best_colors.tolist(), True
            else:
                # Should not happen if local_search is correct, but just in case
                return None, False
        return None, False

    best_n = 0
    best_colors = None
    
    # Try n from 34 down to 10
    # Allocate time based on n? 
    # 34 is the target. Give it most time.
    # 33, 32... give less.
    
    time_per_n = {
        34: 60,
        33: 20,
        32: 10,
        31: 5,
        30: 5,
        29: 5,
        28: 5,
        27: 5,
        26: 5,
        25: 5,
        24: 5,
        23: 5,
        22: 5,
        21: 5,
        20: 5,
        19: 5,
        18: 5,
        17: 5,
        16: 5,
        15: 5,
        14: 5,
        13: 5,
        12: 5,
        11: 5,
        10: 5
    }
    
    for n in range(34, 9, -1):
        if time.time() > deadline - 5:
            break
            
        t_budget = time_per_n.get(n, 5)
        colors, valid = try_n(n, t_budget, seed_offset=n*1000)
        
        if valid:
            best_n = n
            best_colors = colors
            break # Found the largest n that works in our search
    
    # If we didn't find anything, we must return a valid coloring.
    # We can try a very short search for n=10.
    if best_colors is None:
        if time.time() < deadline - 2:
            colors, valid = try_n(10, 10.0, seed_offset=99999)
            if valid:
                best_n = 10
                best_colors = colors
    
    # Absolute fallback: Hardcode a known valid small coloring?
    # Or just return n=1 with [0]? 
    # R(4,5;3) > 1 is trivially true.
    # But the task says "Maximize n".
    # If we can't find a valid one, we return the best we found.
    # But if best_colors is None, we must return something.
    
    if best_colors is None:
        # n=1
        best_n = 1
        best_colors = [0]
        
    return {'n': best_n, 'colors': best_colors}

if __name__ == '__main__':
    res = construct()
    print(f"Found n={res['n']}")
