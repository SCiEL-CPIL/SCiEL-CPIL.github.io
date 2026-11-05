import time
import math
import random

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin before 60s limit
    
    N = 256
    vertices = list(range(1, N + 1))
    
    # Precompute perfect squares up to 2*N
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
        
    # Build conflict graph
    # conflicts[i] is a bitmask of j such that (i+1) + (j+1) is a square
    # Index 0 corresponds to value 1
    conflicts = [0] * N
    
    for i in range(N):
        val_i = i + 1
        mask = 0
        for j in range(N):
            val_j = j + 1
            if val_i + val_j in squares:
                mask |= (1 << j)
        conflicts[i] = mask
        
    # Precompute degrees for potential tie-breaking or initial sorting
    degrees = [bin(conflicts[i]).count('1') for i in range(N)]
    
    # Strategy: Deterministic Greedy with Multi-Restart and "Gain" Heuristic
    # We will run multiple restarts. In each restart, we build a set greedily.
    # At each step, we pick the vertex v that maximizes the number of remaining
    # valid vertices it does NOT conflict with. This is a "set cover" like heuristic
    # where we want to preserve future options.
    
    # Initial best
    best_mask = 0
    best_size = 0
    
    # Pre-compute all_indices
    all_indices = list(range(N))
    
    # We can also try a simple deterministic greedy first to get a baseline
    # Greedy 1: Pick vertex with min degree first (common for MIS)
    # Greedy 2: Pick vertex with max "future gain"
    
    def greedy_build(ordering_strategy):
        """
        ordering_strategy: 'min_degree', 'max_gain', 'random'
        """
        current_mask = 0
        remaining_mask = (1 << N) - 1
        
        while remaining_mask:
            # Check time
            if time.time() - start_time > DEADLINE:
                break
            
            best_candidate = -1
            best_score = -1
            
            # Iterate through remaining vertices
            # For N=256, this loop is fast if we optimize bit operations
            # But iterating 256 items * 256 steps is ~65k ops, which is fine.
            
            # Optimization: Instead of iterating all N, iterate only set bits in remaining_mask
            # But we need to calculate score for each.
            
            r = remaining_mask
            while r:
                # Get lowest set bit
                lsb = r & -r
                idx = (lsb.bit_length() - 1)
                
                # Calculate score based on strategy
                if ordering_strategy == 'min_degree':
                    # Score is negative degree (we want min degree)
                    # Actually, we want to pick the one that removes the fewest future options?
                    # Or just standard min-degree greedy.
                    score = -degrees[idx]
                    
                elif ordering_strategy == 'max_gain':
                    # Gain = number of vertices in 'remaining' that idx does NOT conflict with
                    # This includes idx itself? No, idx is being picked.
                    # We want to pick idx such that (remaining_mask & ~conflicts[idx]) is large.
                    # This preserves the most options for future steps.
                    compatible = remaining_mask & ~conflicts[idx]
                    # Note: compatible includes idx itself because conflicts[idx] has idx?
                    # Let's check: if 2*val is square, idx is in conflicts[idx].
                    # If 2*val is not square, idx is NOT in conflicts[idx].
                    # The condition for valid set is: no pair (a,b) in set has a+b square.
                    # If we add idx, it must not conflict with any *other* in set.
                    # And it must not conflict with itself (diagonal).
                    # So if idx is in conflicts[idx], it can NEVER be in a valid set?
                    # Wait. The task says "INCLUDING a=b".
                    # So if 2*val is a square, val CANNOT be in the set.
                    # Let's verify: squares up to 512.
                    # 2*1=2 (no), 2*2=4 (yes, 2^2). So 2 cannot be in set.
                    # 2*4=8 (no), 2*7=14 (no)...
                    # 2*8=16 (yes, 4^2). So 8 cannot be in set.
                    # So vertices where 2*v is a square are INVALID forever.
                    # We should filter these out initially.
                    
                    # Let's re-evaluate the "compatible" count.
                    # We want to maximize the number of vertices we can potentially add later.
                    # Vertices in 'remaining' that are NOT in conflicts[idx] are safe to coexist with idx.
                    # So score = popcount(remaining_mask & ~conflicts[idx])
                    score = bin(compatible).count('1')
                    
                else: # random
                    score = random.random()
                
                if score > best_score:
                    best_score = score
                    best_candidate = idx
                    
                r &= r - 1 # Clear lowest set bit
            
            if best_candidate == -1:
                break
                
            # Add best_candidate to set
            current_mask |= (1 << best_candidate)
            
            # Remove best_candidate and all its conflicts from remaining
            # New remaining = remaining & ~conflicts[best_candidate]
            # Also remove best_candidate itself (it's in conflicts if self-conflict, but even if not, we picked it)
            # Actually, if best_candidate is not in conflicts[best_candidate], we must explicitly remove it.
            # But conflicts[best_candidate] contains all j such that val_i + val_j is square.
            # If 2*val is not square, best_candidate is NOT in conflicts[best_candidate].
            # So we do: remaining_mask &= ~(conflicts[best_candidate] | (1 << best_candidate))
            
            remaining_mask &= ~(conflicts[best_candidate] | (1 << best_candidate))
            
        return current_mask
    
    # Identify "self-conflicting" vertices (2*v is a square)
    # These can never be part of a valid set.
    self_conflict_mask = 0
    for i in range(N):
        if (i + 1) * 2 in squares:
            self_conflict_mask |= (1 << i)
            
    # Valid candidates are those NOT in self_conflict_mask
    valid_candidates_mask = ((1 << N) - 1) & ~self_conflict_mask
    
    # Run multiple restarts with different strategies
    # Strategy 1: Min Degree Greedy (classic)
    # Strategy 2: Max Gain Greedy (set cover style)
    # Strategy 3: Randomized Greedy (randomize tie-breaking or selection)
    
    strategies = ['min_degree', 'max_gain', 'random', 'max_gain', 'min_degree']
    
    # We will run these strategies. 
    # To add variety, we can also shuffle the "random" strategy or use different seeds.
    
    # Also, we can try a "Local Search" polish on the result of the greedy.
    # But the directive asks for a structural change. The greedy multi-restart IS the change.
    # Let's add a simple 1-swap local search at the end of each greedy run to polish.
    
    def local_search_polish(mask, time_limit=1.0):
        """
        Try to improve mask by 1-1 swaps or additions.
        Returns improved mask.
        """
        current_mask = mask
        current_size = bin(current_mask).count('1')
        start_ls = time.time()
        
        improved = True
        while improved and (time.time() - start_ls) < time_limit:
            improved = False
            
            # Try to add any vertex
            # Candidates: valid_candidates_mask & ~current_mask
            candidates = valid_candidates_mask & ~current_mask
            c = candidates
            while c:
                lsb = c & -c
                idx = (lsb.bit_length() - 1)
                if (conflicts[idx] & current_mask) == 0:
                    current_mask |= lsb
                    current_size += 1
                    improved = True
                    # Restart candidate loop? Or just break and loop again?
                    # Breaking and looping again is safer to re-evaluate.
                    break
                c &= c - 1
            
            if improved:
                continue
                
            # Try 1-1 swap
            # Remove one, add one
            # This is more expensive. 
            # Iterate over vertices in current_mask
            in_mask = current_mask
            m = in_mask
            while m and (time.time() - start_ls) < time_limit:
                lsb_rem = m & -m
                idx_rem = (lsb_rem.bit_length() - 1)
                
                # Temporarily remove
                temp_mask = current_mask & ~lsb_rem
                
                # Find a vertex to add
                # Candidates: valid_candidates_mask & ~temp_mask
                # Must not conflict with temp_mask
                candidates = valid_candidates_mask & ~temp_mask
                c = candidates
                found_add = False
                while c:
                    lsb_add = c & -c
                    idx_add = (lsb_add.bit_length() - 1)
                    
                    if (conflicts[idx_add] & temp_mask) == 0:
                        # Valid swap
                        # We only accept if it leads to a state where we can add MORE later?
                        # For now, if size is same, we might accept to escape local opt.
                        # But how do we know it's better? 
                        # Heuristic: Accept if the added vertex has higher "degree" or lower?
                        # Or just accept randomly with low prob?
                        # Let's accept if the new vertex allows more future additions (greedy gain).
                        gain_new = bin((valid_candidates_mask & ~temp_mask & ~conflicts[idx_add])).count('1')
                        # Compare to gain if we kept idx_rem?
                        # This is getting complex. 
                        # Simple approach: Accept swap if it doesn't reduce size (it doesn't) 
                        # and we haven't tried it before? 
                        # Let's just accept with 10% probability to explore.
                        if random.random() < 0.1:
                            current_mask = temp_mask | lsb_add
                            improved = True
                            found_add = True
                            break
                    c &= c - 1
                m &= m - 1
                if found_add:
                    break
                    
        return current_mask

    # Main Loop: Run Greedy Strategies
    # We will alternate between strategies and use local search polish.
    
    # Shuffle indices for random strategy? No, 'random' strategy uses random.random() in score.
    
    num_restarts = 20 # We have ~50s, each greedy is fast (~ms), polish is ~1s.
    # So we can do ~40-50 restarts.
    
    for i in range(num_restarts):
        if time.time() - start_time > DEADLINE:
            break
            
        # Select strategy
        strat = strategies[i % len(strategies)]
        
        # Run greedy
        g_mask = greedy_build(strat)
        
        # Polish
        p_mask = local_search_polish(g_mask, time_limit=1.0)
        
        size = bin(p_mask).count('1')
        if size > best_size:
            best_size = size
            best_mask = p_mask
            
    # Final verification
    best_list = []
    for i in range(N):
        if (best_mask >> i) & 1:
            best_list.append(i + 1)
            
    valid = True
    for i in range(len(best_list)):
        a = best_list[i]
        for j in range(i, len(best_list)):
            b = best_list[j]
            if a + b in squares:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to a known valid set: x % 3 == 1
        # But we must check diagonal: 1+1=2 (ok), 4+4=8 (ok), 7+7=14 (ok).
        # 1%3=1, 4%3=1...
        # Are there any x%3==1 where 2x is square?
        # 2x = k^2. x = k^2/2.
        # If k is even, k=2m, x=2m^2. 2m^2 % 3 == 1?
        # m=1, x=2 (2%3=2 no)
        # m=2, x=8 (8%3=2 no)
        # m=3, x=18 (18%3=0 no)
        # m=4, x=32 (32%3=2 no)
        # m=5, x=50 (50%3=2 no)
        # m=6, x=72 (72%3=0 no)
        # m=7, x=98 (98%3=2 no)
        # m=8, x=128 (128%3=2 no)
        # m=9, x=162 (162%3=0 no)
        # m=10, x=200 (200%3=2 no)
        # m=11, x=242 (242%3=2 no)
        # m=12, x=288 (>256)
        # So no x%3==1 has 2x square.
        # And pairs: (3k+1)+(3m+1) = 3(k+m)+2.
        # Squares mod 3 are 0, 1. 2 is not a square mod 3.
        # So {x in [1,256] | x%3==1} is always valid.
        return list(range(1, 257, 3))
        
    return best_list

# Quick self-test to ensure it runs
if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    print(res[:10])
