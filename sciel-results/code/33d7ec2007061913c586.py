import random
import time

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry.
    """
    best_result = None
    best_scope = float('inf')
    
    # Time limit check
    start_time = time.time()
    time_limit = 110.0  # Leave some buffer for the evaluator
    
    # Number of restarts
    max_restarts = 2000
    
    # Beam width / candidate pool size for randomized greedy
    # We consider the top 'pool_size' smallest valid candidates for each step
    pool_size = 10
    
    for _ in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        used_diffs = set()
        rows = []
        
        success = True
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                last_val = row[-1]
                valid_candidates = []
                
                # We need to find x > last_val
                # The maximum possible value is bounded by the current best_scope
                # If we haven't found a solution yet, we don't have a tight bound, 
                # but we can use a heuristic upper bound.
                # The counting bound is 105. The known best is 112.
                # Let's search up to current_best_scope - 1 if available, else a generous bound.
                upper_bound = best_scope - 1 if best_scope < float('inf') else 200
                
                for x in range(last_val + 1, upper_bound + 1):
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used_diffs:
                        valid_candidates.append(x)
                        # Optimization: if we have enough candidates, we can stop early?
                        # No, we want the smallest ones. But if we are randomizing, 
                        # we might want to see a few more? 
                        # Actually, for the "top K" strategy, we just need the K smallest.
                        # So we can break if we have pool_size candidates?
                        # Yes, because we only pick from the smallest pool_size.
                        if len(valid_candidates) >= pool_size:
                            break
                
                if not valid_candidates:
                    success = False
                    break
                
                # Select a candidate from the top pool_size
                # Weighted random selection favoring smaller values
                # Or simply pick a random one from the first pool_size
                # Let's use a simple random choice from the first pool_size to encourage exploration
                # But to stay efficient, we should bias towards smaller ones.
                # Let's pick index 0 with high prob, and others with lower prob.
                # Or just random.choice(valid_candidates[:pool_size])
                
                # To balance exploration and exploitation:
                # 50% chance pick the smallest (greedy)
                # 50% chance pick random from top pool_size
                if random.random() < 0.5:
                    x = valid_candidates[0]
                else:
                    x = random.choice(valid_candidates[:pool_size])
                
                row.append(x)
                used_diffs |= {x - a for a in row[:-1]} # Wait, row[:-1] is the old row
                # Actually, new_diffs was calculated as {x - a for a in row} where row was the OLD row.
                # So we can just do:
                # used_diffs |= new_diffs
                # But we didn't store new_diffs. Let's recalculate or store it.
                # Better to store new_diffs in the loop above.
                
                # Let's restructure the inner loop to store new_diffs.
                
            if not success:
                break
            
            rows.append(row)
            # The scope is the max of all elements in all rows.
            # Since rows are increasing, the max of a row is its last element.
            # Since we build rows sequentially, the global max is the max of the last elements.
            # However, a later row might have a larger max than an earlier one.
            # The score is the largest entry in the ENTIRE set.
            current_scope = max(r[-1] for r in rows)
            
            if len(rows) == n_rows:
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_result = [r[:] for r in rows]
                    # If we hit the theoretical lower bound, we can stop
                    # Counting bound: n_rows * k*(k+1)/2 differences.
                    # Total diffs needed = n_rows * k*(k+1)/2.
                    # Max diff is scope. So scope >= sqrt(2 * total_diffs) approx?
                    # Actually, the differences must be distinct integers from 1 to scope.
                    # So we need at least n_rows * k*(k+1)/2 distinct differences.
                    # Thus scope >= n_rows * k*(k+1)/2.
                    # For n_rows=7, k=5 (wait, task says k=5? "7 rows of 6 increasing integers" -> k=5 in the formula a_0..a_k? No, "k+1 integers". So if 6 integers, k=5.
                    # The prompt says "construct(n_rows, k)". And "list of k+1 integers".
                    # So if the evaluator passes k=5, we need 6 integers.
                    # Let's assume the evaluator passes the correct k.
                    # For 7 rows of 6 integers (5 gaps per row, 15 diffs per row), total diffs = 105.
                    # So scope must be at least 105.
                    if best_scope <= 105:
                        break
        
        # End for _ in range(n_rows)
        
    # End for _ in range(max_restarts)
    
    if best_result is None:
        # Fallback to deterministic greedy if random search failed (unlikely)
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
        
    return best_result
