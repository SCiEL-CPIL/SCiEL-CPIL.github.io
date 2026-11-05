import itertools
import random
import time
import heapq
import math

def construct(v, k, t):
    # Parameters for the specific task are v=24, k=7, t=3
    # But we keep it general for robustness if possible, though optimized for this case.
    
    all_subs = list(itertools.combinations(range(v), t))
    sub_to_idx = {s: i for i, s in enumerate(all_subs)}
    n_subs = len(all_subs)
    
    # Precompute the list of t-subsets for any given k-subset (block)
    # To speed up, we can precompute the triples for a block if we generate many, 
    # but blocks are dynamic. We'll use a helper.
    
    def get_block_triples(block):
        return list(itertools.combinations(block, t))

    def get_block_triple_indices(block):
        return [sub_to_idx[tr] for tr in get_block_triples(block)]

    # --- Phase 1: Greedy Construction ---
    # Improved Greedy: 
    # 1. Start with empty set of blocks.
    # 2. Maintain set of uncovered triples.
    # 3. While uncovered:
    #    a. Pick a random uncovered triple T.
    #    b. Construct a block B containing T.
    #       - Start with T.
    #       - Greedily add points to B to maximize the number of *other* uncovered triples covered.
    #    c. Add B to blocks.
    #    d. Update uncovered.
    
    def greedy_construct(seed):
        random.seed(seed)
        uncovered = set(range(n_subs))
        blocks = []
        
        # To speed up gain calculation, we can precompute for each point x and each subset S of size t-1
        # the list of triples (S, x). But S changes.
        # Let's just iterate.
        
        while uncovered:
            # Pick a random uncovered triple
            triple_idx = random.choice(list(uncovered))
            triple = all_subs[triple_idx]
            
            # Build a block containing this triple
            # Start with the triple
            block = list(triple)
            
            # Greedily add points to maximize coverage of uncovered triples
            while len(block) < k:
                best_x = -1
                max_gain = -1
                best_gain_ties = []
                
                # Candidates: points not in block
                for x in range(v):
                    if x in block:
                        continue
                    
                    # Gain = number of uncovered triples formed by adding x to block
                    # Triples formed: (t-1) elements from block + x
                    gain = 0
                    for rest in itertools.combinations(block, t-1):
                        new_triple = tuple(sorted(rest + (x,)))
                        idx = sub_to_idx[new_triple]
                        if idx in uncovered:
                            gain += 1
                    
                    if gain > max_gain:
                        max_gain = gain
                        best_x = x
                        best_gain_ties = [x]
                    elif gain == max_gain:
                        best_gain_ties.append(x)
                        
                if best_x == -1:
                    # If no gain, pick random
                    candidates = [x for x in range(v) if x not in block]
                    if candidates:
                        best_x = random.choice(candidates)
                    else:
                        break
                else:
                    # Pick randomly among ties to diversify
                    best_x = random.choice(best_gain_ties)
                    
                block.append(best_x)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for tr in get_block_triples(block):
                idx = sub_to_idx[tr]
                uncovered.discard(idx)
                
        return blocks

    # --- Phase 2: Local Search (Simulated Annealing / Hill Climbing) ---
    
    def local_search(initial_blocks, time_limit):
        current_blocks = [b[:] for b in initial_blocks]
        current_count = len(current_blocks)
        
        # Maintain coverage counts
        cover_counts = [0] * n_subs
        for b in current_blocks:
            for tr in get_block_triples(b):
                cover_counts[sub_to_idx[tr]] += 1
        
        # Verify validity
        if any(c == 0 for c in cover_counts):
            return None, float('inf') # Invalid
            
        best_blocks = [b[:] for b in current_blocks]
        best_count = current_count
        
        start_time = time.time()
        
        # Simulated Annealing Parameters
        # We will use a temperature-based acceptance for swaps that don't reduce count
        # but might improve "slack".
        
        # Slack metric: Number of triples with count == 1 (critical).
        # Lower slack is better (more redundancy).
        
        def count_critical(cover_counts):
            return sum(1 for c in cover_counts if c == 1)
        
        current_slack = count_critical(cover_counts)
        best_slack = current_slack
        
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.999
        
        no_improve_count = 0
        
        while time.time() - start_time < time_limit:
            
            # --- Move 1: Try to Remove a Redundant Block ---
            removed = False
            # To make this faster, we can check blocks in random order or maintain a list of "potentially redundant" blocks
            # For now, random scan.
            idxs = list(range(len(current_blocks)))
            random.shuffle(idxs)
            
            for i in idxs:
                b = current_blocks[i]
                redundant = True
                for tr in get_block_triples(b):
                    if cover_counts[sub_to_idx[tr]] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove it
                    for tr in get_block_triples(b):
                        cover_counts[sub_to_idx[tr]] -= 1
                    current_blocks.pop(i)
                    current_count -= 1
                    removed = True
                    current_slack = count_critical(cover_counts)
                    
                    if current_count < best_count:
                        best_count = current_count
                        best_blocks = [b[:] for b in current_blocks]
                        best_slack = current_slack
                        no_improve_count = 0
                    else:
                        # Count didn't change? Impossible if we removed.
                        pass
                    break
                    
            if removed:
                continue # Restart loop to check for more removals
            
            # --- Move 2: Try a Swap (Replace a block) ---
            if not current_blocks:
                break
                
            i = random.randint(0, len(current_blocks) - 1)
            b_old = current_blocks[i]
            
            # Identify critical triples in b_old (those with count == 1)
            critical_triples = []
            for tr in get_block_triples(b_old):
                if cover_counts[sub_to_idx[tr]] == 1:
                    critical_triples.append(tr)
            
            if not critical_triples:
                # This is a redundant block. Remove it.
                for tr in get_block_triples(b_old):
                    cover_counts[sub_to_idx[tr]] -= 1
                current_blocks.pop(i)
                current_count -= 1
                current_slack = count_critical(cover_counts)
                
                if current_count < best_count:
                    best_count = current_count
                    best_blocks = [b[:] for b in current_blocks]
                    best_slack = current_slack
                    no_improve_count = 0
                continue
            
            # We need to find a new block b_new such that:
            # 1. It contains all points in the union of critical triples? No, it must CONTAIN the critical triples.
            #    i.e., for each tr in critical_triples, tr is a subset of b_new.
            
            U = set()
            for tr in critical_triples:
                U.update(tr)
                
            if len(U) > k:
                # Impossible to cover all critical triples with one block
                # Skip this swap
                continue
            
            # We need to pick k - len(U) additional points from the remaining v - len(U) points.
            available_points = [p for p in range(v) if p not in U]
            remaining_slots = k - len(U)
            
            # Sample a few candidate new blocks
            found_new_block = False
            best_candidate = None
            best_candidate_slack = float('inf')
            
            # Try multiple random completions
            for _ in range(10):
                if remaining_slots > len(available_points):
                    break
                extra_points = random.sample(available_points, remaining_slots)
                b_new = sorted(list(U) + extra_points)
                
                # Check if b_new is valid (distinct elements, size k) - it is by construction.
                # Check if b_new covers all critical triples.
                # Since U is the union of points in critical triples, and b_new contains U,
                # does b_new contain the critical triples?
                # Yes.
                
                # Calculate the change in slack
                # We need to simulate the swap to calculate new slack.
                # This is expensive. 
                # Approximation: 
                # Old slack contribution from b_old's triples:
                # Triples in b_old that are critical (count 1) will become count 0 if we remove b_old.
                # But b_new will cover them, so they become count 1 again (covered by b_new).
                # So their status remains "critical" (count 1).
                # Triples in b_old that are NOT critical (count > 1) will have count decreased by 1.
                #   If they were count 2, they become count 1 (NEW critical).
                #   If they were count > 2, they remain non-critical.
                # Triples in b_new that are NOT in b_old:
                #   If they were uncovered (count 0)? Impossible, b_new is valid only if it covers criticals, 
                #   but it might cover other triples. 
                #   Wait, b_new is a valid block. It covers C(k,t) triples.
                #   Some of these triples might have been covered by b_old (and others).
                #   Some might have been covered by other blocks.
                #   Some might have been uncovered? No, the covering is valid, so all triples are covered.
                #   So all triples in b_new have count >= 1.
                #   When we add b_new, their count increases by 1.
                #   If they were count 1, they become count 2 (LOSE critical).
                #   If they were count > 1, they remain non-critical.
                
                # So, Delta Slack = 
                #   (Number of triples in b_old \ b_new with count == 2)  [Become Critical]
                #   - (Number of triples in b_new \ b_old with count == 1) [Lose Critical]
                #   Note: Triples in intersection b_old ∩ b_new:
                #   - If critical (count 1): Removed from b_old, Added to b_new. Count stays 1. No change.
                #   - If not critical (count > 1): Removed from b_old (count-1), Added to b_new (count+1). Net 0.
                
                # Let's calculate this delta.
                
                # 1. Identify triples in b_old
                trs_old = set(get_block_triples(b_old))
                # 2. Identify triples in b_new
                trs_new = set(get_block_triples(b_new))
                
                # Triples only in b_old
                only_old = trs_old - trs_new
                # Triples only in b_new
                only_new = trs_new - trs_old
                
                delta_slack = 0
                for tr in only_old:
                    idx = sub_to_idx[tr]
                    if cover_counts[idx] == 2:
                        delta_slack += 1
                        # If it becomes 1, it is critical.
                
                for tr in only_new:
                    idx = sub_to_idx[tr]
                    if cover_counts[idx] == 1:
                        delta_slack -= 1
                        # If it becomes 2, it is no longer critical.
                
                new_slack = current_slack + delta_slack
                
                if new_slack < best_candidate_slack:
                    best_candidate_slack = new_slack
                    best_candidate = b_new
                    best_candidate_delta = delta_slack
                
            if best_candidate is not None:
                # Acceptance criterion
                # If slack decreases, accept.
                # If slack increases, accept with probability exp(-delta / T)
                # If slack same, accept with some probability to allow exploration.
                
                if best_candidate_delta < 0:
                    accept = True
                elif best_candidate_delta == 0:
                    accept = random.random() < 0.5
                else:
                    accept = random.random() < math.exp(-best_candidate_delta / T)
                
                if accept:
                    # Perform Swap
                    for tr in get_block_triples(b_old):
                        cover_counts[sub_to_idx[tr]] -= 1
                    for tr in get_block_triples(best_candidate):
                        cover_counts[sub_to_idx[tr]] += 1
                    
                    current_blocks[i] = best_candidate
                    current_slack += best_candidate_delta
                    found_new_block = True
                    
                    if current_count < best_count:
                        best_count = current_count
                        best_blocks = [b[:] for b in current_blocks]
                        best_slack = current_slack
                        no_improve_count = 0
                    elif current_slack < best_slack:
                        best_slack = current_slack
                        best_blocks = [b[:] for b in current_blocks]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
            
            if not found_new_block:
                no_improve_count += 1
            
            # Cooling
            T *= cooling_rate
            if T < T_min:
                T = T_min
                # Reheat if stuck
                if no_improve_count > 1000:
                    T = 1.0
                    no_improve_count = 0
                    
        return best_blocks, best_count

    # --- Main Execution ---
    
    start_time = time.time()
    total_time_limit = 110 # Leave 10s for safety
    
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple iterations with different random seeds
    seed = 42
    while time.time() - start_time < total_time_limit:
        # Greedy
        blocks = greedy_construct(seed)
        
        # Local Search
        # Allocate remaining time, but reserve a bit for next greedy if possible?
        # Simpler: Run LS for a fixed chunk or until time runs out.
        # Since we run multiple seeds, we should limit LS time per seed.
        # Let's say 10 seconds per LS run.
        ls_blocks, ls_count = local_search(blocks, time_limit=min(10, total_time_limit - (time.time() - start_time)))
        
        if ls_blocks is not None and ls_count < best_count:
            best_count = ls_count
            best_blocks = ls_blocks[:]
            
        seed += 1
        
        # If we are getting close to the lower bound or time is running out, break
        if best_count <= 70: # Lower bound is 69
            break
            
    if best_blocks is None:
        # Fallback: return the last greedy result
        blocks = greedy_construct(seed)
        return blocks
        
    return best_blocks
