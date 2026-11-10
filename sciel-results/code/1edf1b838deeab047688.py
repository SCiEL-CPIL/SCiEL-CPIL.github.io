"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit
    
    # We want to maximize n. Published construction is 34.
    # Let's try to find valid colorings for n in descending order from 20 down to 10?
    # Or ascending?
    # Local search is better if we start with a reasonable n and try to fix violations.
    # If we start with n=15, we might find a valid one quickly.
    # If we start with n=30, it's a huge space.
    
    # Strategy:
    # 1. Try increasing n from 10 up to 34.
    # 2. For each n, run local search with a time budget.
    # 3. If we find a valid coloring for n, save it and try n+1.
    # 4. If time runs out, return the best n found.
    
    best_n = 9
    best_colors = None
    
    # Pre-compute all triples for max n=34? 
    # C(34,3) = 5984. C(40,3) = 9880. Small enough to handle.
    # But checking validity is expensive if done naively.
    
    # Let's define a helper to check validity efficiently?
    # Or just count violations during search.
    
    def get_triples(n):
        return list(combinations(range(n), 3))
    
    # We'll iterate n from 10 to 34
    # To be safe and fast, let's start from a known small n and go up.
    # Actually, let's try a few random restarts for a fixed n, say n=20, then 25, then 30.
    # Or just a simple loop:
    
    # Let's try n from 12 to 34.
    # For each n, we have a time slice.
    # Total time 110s. 23 values of n. ~4-5s per n.
    
    for n in range(12, 35):
        if time.time() > deadline - 5.0:
            break
            
        # Time budget for this specific n
        time_budget_for_n = 3.0 # seconds
        n_start = time.time()
        
        # Initialize random coloring
        triples = get_triples(n)
        num_triples = len(triples)
        
        # Map triple to index for fast access
        # We can just use the list index if we iterate carefully, 
        # but flipping requires knowing which triples contain a vertex?
        # Actually, local search usually flips one edge (triple) at a time.
        
        # To evaluate change in objective function efficiently:
        # Objective: (# Red K4s) + (# Blue K5s)
        # Flipping a triple (a,b,c) from Red to Blue:
        # - It might break Red K4s containing (a,b,c).
        # - It might create Blue K5s containing (a,b,c).
        
        # This requires maintaining counts of sub-hyperedges.
        # For Red K4: A K4 is 4 vertices. It has 4 triples.
        # If we flip one triple in a K4 to Blue, the K4 is no longer Red.
        # So we need to know for each triple, how many Red K4s it is part of.
        # And for Blue K5: A K5 is 5 vertices. It has 10 triples.
        # If we flip one triple in a K5 to Red, the K5 is no longer Blue.
        
        # Let's maintain:
        # red_k4_count[t] : number of Red K4s containing triple t
        # blue_k5_count[t] : number of Blue K5s containing triple t
        
        # Initialization is O(n^4) for K4s and O(n^5) for K5s?
        # n=30: C(30,4)=27405, C(30,5)=142506.
        # For each K4, check if all 4 triples are Red.
        # For each K5, check if all 10 triples are Blue.
        # This is feasible for n=30 within a few seconds.
        
        # Let's try this approach for n up to 30.
        
        # 1. Generate random colors
        colors = [random.randint(0, 1) for _ in range(num_triples)]
        
        # 2. Compute initial violation counts
        # We need to map vertices to triples? Or just iterate all K4s and K5s.
        
        # To make flipping fast, we precompute:
        # For each triple index t, list of K4 indices containing it?
        # And for each triple index t, list of K5 indices containing it?
        
        # This is memory intensive but n is small.
        # C(30,3) ~ 4000 triples.
        # Each triple is in C(27,1)=27 K4s? No, a triple (a,b,c) is in K4s with any other vertex d.
        # So 27 K4s.
        # Each triple is in C(25,2)=300 K5s? No, a triple (a,b,c) is in K5s with any 2 other vertices d,e.
        # C(n-3, 2) = C(27,2) = 351 K5s.
        
        # Total entries in adjacency lists:
        # K4s: 4000 * 27 ~ 100k
        # K5s: 4000 * 351 ~ 1.4M
        # This is fine.
        
        # However, building these lists takes time.
        # Let's see if we can do it within the time budget.
        
        try:
            # Build mapping: triple_idx -> list of k4_indices, triple_idx -> list of k5_indices
            # First, generate all K4s and K5s
            k4s = list(combinations(range(n), 4))
            k5s = list(combinations(range(n), 5))
            
            # Map vertex tuples to triple indices?
            # Triple (a,b,c) is identified by sorted tuple.
            triple_to_idx = {tuple(t): i for i, t in enumerate(triples)}
            
            # For each K4, find its 4 triples
            # For each K5, find its 10 triples
            
            # Initialize counts
            red_k4_count = [0] * num_triples
            blue_k5_count = [0] * num_triples
            
            # Also store the K4s and K5s to check validity later?
            # We just need the total count of violations.
            # Total Violations = sum(1 for k4 in k4s if all red) + sum(1 for k5 in k5s if all blue)
            
            # To update counts on flip:
            # If we flip triple t from Red to Blue:
            # - For each K4 containing t: if the other 3 triples were Red, this K4 was Red. Now it's not. 
            #   So we decrement red_k4_count for those 3 other triples? No.
            #   We need to know if the K4 becomes non-Red.
            #   It becomes non-Red if it WAS Red.
            #   It WAS Red if all 4 triples were Red.
            #   Since we are flipping t (Red->Blue), the K4 is no longer Red.
            #   So if it was Red, we reduce the "Red K4 status" for that K4.
            #   This affects the counts of the OTHER 3 triples in that K4.
            #   Specifically, red_k4_count[other_t] should be decremented by 1 for each such K4.
            
            # Similarly for Blue K5s.
            
            # This requires knowing the state of the other triples.
            # This is getting complex to implement correctly and fast in Python within 3s per n.
            
            # Alternative: Simple Simulated Annealing or Hill Climbing with full re-evaluation?
            # Full re-evaluation for n=30:
            # Check all C(30,4)=27405 K4s.
            # Check all C(30,5)=142506 K5s.
            # Total ~ 170k checks.
            # Each check is O(1) if we have a fast way to check "all red" or "all blue".
            # We can precompute for each K4 the indices of its 4 triples.
            # And for each K5 the indices of its 10 triples.
            
            # Let's try this simpler approach. It might be slow in Python but let's see.
            
            # Precompute K4 triple indices
            k4_triples = []
            for k4 in k4s:
                ts = []
                for i in range(4):
                    t = tuple(sorted(k4[:i] + k4[i+1:]))
                    ts.append(triple_to_idx[t])
                k4_triples.append(ts)
                
            # Precompute K5 triple indices
            k5_triples = []
            for k5 in k5s:
                ts = []
                for combo in combinations(range(5), 3):
                    t = tuple(sorted([k5[i] for i in combo]))
                    ts.append(triple_to_idx[t])
                k5_triples.append(ts)
            
            # Function to count violations
            def count_violations(cols):
                v = 0
                # Red K4s
                for ts in k4_triples:
                    if cols[ts[0]] == 0 and cols[ts[1]] == 0 and cols[ts[2]] == 0 and cols[ts[3]] == 0:
                        v += 1
                # Blue K5s
                for ts in k5_triples:
                    if all(c == 1 for c in [cols[ts[i]] for i in range(10)]):
                        v += 1
                return v
            
            # Local Search
            current_v = count_violations(colors)
            best_v = current_v
            best_cols = colors[:]
            
            no_improve = 0
            max_no_improve = 500
            
            while time.time() < n_start + time_budget_for_n and no_improve < max_no_improve:
                # Pick a random triple to flip
                idx = random.randint(0, num_triples - 1)
                old_color = colors[idx]
                colors[idx] = 1 - old_color
                
                new_v = count_violations(colors)
                
                if new_v < best_v:
                    best_v = new_v
                    best_cols = colors[:]
                    no_improve = 0
                else:
                    # Revert if not better? Or keep if equal?
                    # Let's revert if worse, keep if equal?
                    # Standard hill climbing: revert if worse.
                    if new_v > current_v:
                        colors[idx] = old_color
                        no_improve += 1
                    else:
                        # Equal or better (but not better than best)
                        # If equal, we might get stuck in plateaus.
                        # Let's allow moves that don't increase violations.
                        current_v = new_v
                        no_improve += 1
                        
                if best_v == 0:
                    break
                    
            if best_v == 0:
                best_n = n
                best_colors = best_cols
                # Don't break, try to go higher?
                # The loop continues to n+1.
                
        except Exception:
            pass
            
    # If we didn't find any valid n >= 12, return the seed solution or n=9
    if best_n < 12:
        n = 9
        return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
    else:
        return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END
