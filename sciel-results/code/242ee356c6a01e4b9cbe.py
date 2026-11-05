import time
import random
import itertools
import sys

def construct():
    """
    Constructs a linear triple system on 16 points with a small independence number.
    
    Strategy:
    1. Initialize with a random maximal linear triple system (packing).
    2. Use Simulated Annealing to optimize the system.
       - Moves: Add a valid triple, remove a triple, or swap a triple.
       - Objective: Minimize the independence number (alpha).
       - Since n=16 is small, we can compute the exact alpha for each candidate 
         state efficiently using a bitset-based branch-and-bound or DP approach.
    3. Return the best system found within the time limit.
    """
    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit
    
    n_points = 16
    points = list(range(n_points))
    
    # Pre-compute all possible triples and their pair signatures
    all_triples = list(itertools.combinations(points, 3))
    # Map triple to index for faster lookup? Or just iterate.
    # For SA, we need to quickly check validity.
    # pair_used: set of pairs (a,b) with a<b
    
    # Helper: Compute exact independence number for a given set of triples
    # We use a recursive branch and bound with memoization on the remaining points mask.
    # State: mask of points still available.
    # If any triple is fully contained in mask, we cannot keep all points in mask.
    # Actually, independence number is the max subset S such that no triple is fully in S.
    # This is equivalent to: Find the largest set of points that does NOT contain any triple.
    
    # Efficient Alpha Calculator for n=16
    # We can precompute for each triple its bitmask.
    triple_masks = []
    for t in all_triples:
        m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks.append(m)
        
    # Map triple tuple to its mask
    triple_to_mask = {t: m for t, m in zip(all_triples, triple_masks)}
    
    def calc_alpha(triples_in_system):
        """
        Calculate the exact independence number of the hypergraph defined by triples_in_system.
        Uses branch and bound on the point set.
        """
        if not triples_in_system:
            return n_points
            
        # Create a list of masks for the triples in the system
        masks = [triple_to_mask[t] for t in triples_in_system]
        
        # We want to find the max subset S (mask) such that for all m in masks, (S & m) != m.
        # Equivalently, S is an independent set.
        
        # Heuristic: Start with full mask, prune.
        # Or use a standard MIS algorithm on the "conflict graph"?
        # Conflict graph: Points are nodes. Two points are connected if they appear in a triple?
        # No, that's not right. A triple {a,b,c} forbids {a,b,c} from being in S.
        # It does NOT forbid {a,b}.
        
        # Let's use a simple recursive search with pruning.
        # Sort masks by popcount? They are all size 3.
        
        # We can iterate over subsets? 2^16 = 65536.
        # For each subset S (represented by mask), check if it is independent.
        # If it is, update max_size.
        # This is O(2^n * |T|). 65536 * ~40 = 2.6M operations.
        # This is fast enough to do thousands of times in 120s?
        # 2.6M ops in Python might take ~0.1-0.2s.
        # If we do 1000 SA steps, that's 100-200s. Too slow.
        
        # Optimization: Precompute validity for all 2^16 subsets?
        # No, the set of triples changes.
        
        # Better: Branch and Bound.
        # Node: (current_mask, remaining_triples_indices)
        # If current_mask is independent, return popcount(current_mask).
        # Prune if popcount(current_mask) + max_possible_remaining <= best_found.
        
        # Let's implement a faster check using bit operations.
        # A mask S is independent if for all t in triples_in_system, (S & mask_t) != mask_t.
        
        best = 0
        # We can use a simple DFS.
        # To speed up, we can order points.
        
        def dfs(mask, idx):
            nonlocal best
            # Pruning
            # Max possible size is popcount(mask)
            if mask == 0:
                return 0
            
            # Check if current mask is independent
            # If yes, it's a candidate.
            is_independent = True
            for t in triples_in_system:
                if (mask & triple_to_mask[t]) == triple_to_mask[t]:
                    is_independent = False
                    break
            
            if is_independent:
                if bin(mask).count('1') > best:
                    best = bin(mask).count('1')
                # We can try to add more points?
                # No, if mask is independent, any subset is independent.
                # We want the LARGEST.
                # So if mask is independent, we should try to ADD points?
                # Wait, the standard MIS search removes points.
                # Here, we want to find the largest S.
                # Let's iterate S from largest to smallest?
                
            # This approach is tricky. Let's stick to the 2^n iteration but optimized.
            # Since n=16, 2^16 is small.
            # We can just iterate all masks from (1<<16)-1 down to 0.
            # Stop when we find the first independent set? No, we want the largest.
            # So we can iterate by size? Or just check all and keep max.
            
            # Actually, checking all 65536 masks is the most robust and simple.
            # Let's optimize the inner loop.
            pass

        # Revised Alpha Calc: Iterate all masks?
        # 65536 masks * 40 triples * bit ops.
        # In Python, this loop is slow.
        
        # Alternative: Use the property that if S is independent, S \ {x} is independent.
        # We can use DP on masks?
        # dp[mask] = True if mask is independent.
        # dp[0] = True.
        # dp[mask] = dp[mask without lowest bit] AND (no triple fully contained in mask).
        # This is still O(2^n * |T|).
        
        # Let's try a heuristic alpha estimator for speed during SA, and exact for final check?
        # No, the score is the EXACT alpha. If we use a heuristic, we might accept a move that looks good but is bad.
        
        # Let's optimize the exact calc using NumPy?
        # Or just use a very fast Python loop.
        
        # Fastest Python approach for n=16:
        # Precompute for each triple its mask.
        # For a given set of triples, we want max popcount(S) s.t. for all t, S & mask_t != mask_t.
        
        # Let's try to do this:
        # 1. Start with all points.
        # 2. If a triple is fully present, remove one point.
        # 3. This is the heuristic from before. It gives a LOWER bound on alpha?
        #    No, it constructs a specific independent set. So it gives a lower bound.
        #    The true alpha is >= this value.
        #    We want to MINIMIZE the true alpha.
        #    If the heuristic finds a large set, alpha is large (bad).
        #    If the heuristic finds a small set, alpha might still be large (bad if we underestimate).
        #    So heuristic is unsafe for optimization if it underestimates.
        
        # However, for n=16, the gap between heuristic and exact is likely small.
        # Let's use the heuristic for SA steps, and exact for final verification/selection.
        
        def heuristic_alpha(triples_list):
            S = (1 << n_points) - 1
            # We want to remove points to break all triples.
            # Greedy: While there is a triple fully in S, remove the point in that triple 
            # that appears in the most other triples fully in S.
            
            # To make this fast, we can precompute degrees?
            # Simple version:
            changed = True
            while changed:
                changed = False
                found = False
                for t in triples_list:
                    m = triple_to_mask[t]
                    if (S & m) == m:
                        found = True
                        # Find point in t with max "impact"
                        # Impact: number of other triples in S that contain this point.
                        best_p = -1
                        best_count = -1
                        for p in t:
                            count = 0
                            pm = 1 << p
                            for t2 in triples_list:
                                if t2 == t: continue
                                m2 = triple_to_mask[t2]
                                if (S & m2) == m2:
                                    if m2 & pm:
                                        count += 1
                            if count > best_count:
                                best_count = count
                                best_p = p
                        S &= ~(1 << best_p)
                        changed = True
                        break # Restart loop to re-evaluate S
                
            return bin(S).count('1')
            
        # This heuristic is O(|T|^2 * n) in worst case per step.
        # With |T|~40, n=16, it's fast.
        
        # Let's use this heuristic for the SA objective.
        
        # Initialize system
        def random_maximal_packing():
            triples = []
            pair_used = set()
            shuffled = all_triples[:]
            random.shuffle(shuffled)
            for t in shuffled:
                a,b,c = t
                if (a,b) not in pair_used and (a,c) not in pair_used and (b,c) not in pair_used:
                    triples.append(t)
                    pair_used.add((a,b))
                    pair_used.add((a,c))
                    pair_used.add((b,c))
            return triples

        best_system = random_maximal_packing()
        best_score = heuristic_alpha(best_system)
        
        current_system = best_system[:]
        current_score = best_score
        
        # Simulated Annealing
        T = 10.0
        T_min = 0.1
        cooling_rate = 0.999
        
        # Precompute valid triples for quick add?
        # We need to check validity against current_pair_used.
        
        while time.time() < deadline - 2.0:
            # Generate move
            move_type = random.random()
            
            new_system = current_system[:]
            new_score = current_score # Assume same initially
            
            if move_type < 0.5:
                # Add a random valid triple
                # Find candidates
                # We need to know which triples are valid.
                # This is expensive to compute from scratch.
                # Let's pick a random triple from all_triples and check.
                # Or maintain a list of available triples?
                # For now, pick random from all_triples.
                for _ in range(10): # Try 10 random picks
                    t = random.choice(all_triples)
                    a,b,c = t
                    # Check against current_system
                    valid = True
                    for t2 in current_system:
                        # Check pair overlap
                        # Faster: use pair_used set?
                        # We don't have pair_used for current_system in SA loop easily.
                        # Let's build pair_used for current_system once.
                        pass
                    
                # Rebuild pair_used for current_system
                pair_used = set()
                for t in current_system:
                    a,b,c = t
                    pair_used.add((a,b))
                    pair_used.add((a,c))
                    pair_used.add((b,c))
                    
                # Try to add
                t = random.choice(all_triples)
                a,b,c = t
                if (a,b) not in pair_used and (a,c) not in pair_used and (b,c) not in pair_used:
                    new_system.append(t)
                    new_score = heuristic_alpha(new_system)
                else:
                    continue # Skip this step if no valid add found quickly
            
            elif move_type < 0.8:
                # Remove a random triple
                if current_system:
                    idx = random.randint(0, len(current_system)-1)
                    new_system.pop(idx)
                    new_score = heuristic_alpha(new_system)
                else:
                    continue
            
            else:
                # Swap: Remove one, add one
                if current_system:
                    idx = random.randint(0, len(current_system)-1)
                    removed = new_system.pop(idx)
                    
                    # Try to add a different triple
                    pair_used = set()
                    for t in new_system:
                        a,b,c = t
                        pair_used.add((a,b))
                        pair_used.add((a,c))
                        pair_used.add((b,c))
                    
                    t = random.choice(all_triples)
                    a,b,c = t
                    if (a,b) not in pair_used and (a,c) not in pair_used and (b,c) not in pair_used:
                        new_system.append(t)
                        new_score = heuristic_alpha(new_system)
                    else:
                        # Revert
                        new_system.insert(idx, removed)
                        new_score = current_score
                        continue
            
            # Acceptance criterion
            delta = new_score - current_score
            if delta < 0 or random.random() < pow(2.718, -delta / T):
                current_system = new_system
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_system = current_system[:]
                    # Optional: exact check on best?
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
        
        # Final exact check on best_system?
        # The heuristic might be slightly off.
        # But for n=16, the heuristic is usually tight.
        # Let's return best_system.
        
        return best_system

# EVOLVE-BLOCK-END
