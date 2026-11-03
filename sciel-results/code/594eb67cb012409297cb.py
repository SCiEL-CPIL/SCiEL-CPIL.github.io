"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    # Deadline guard: must finish well under 150s wall-clock.
    deadline = time.time() + 145.0
    rng = random.Random(42)
    
    # We need n_rows rows, each with k+1 entries starting at 0.
    # Objective: minimize the maximum entry (scope).
    # Constraint: all positive differences within a row must be distinct across all rows.
    
    # Strategy:
    # 1. Build rows greedily one by one, trying to keep the scope small.
    # 2. Use backtracking or restarts if we get stuck or if the scope is growing too fast.
    # 3. Use a "used" set of differences.
    # 4. For each new row, try to build it with the smallest possible max value.
    #    We can do this by trying to place the next smallest number that doesn't conflict.
    #    If we can't, we might need to backtrack or restart the whole thing with a different seed/order.
    
    # A better heuristic:
    # Instead of just greedy, we can try to optimize the order of rows or the specific choices.
    # Since the problem is symmetric with respect to the order of rows (the set of differences is global),
    # the order in which we build rows matters for the greedy choice.
    
    # Let's implement a greedy algorithm with randomization and restarts.
    # We will try to build the set of rows.
    # We keep track of the global 'used' differences.
    
    def build_greedy(n_rows, k, rng, target_scope=None):
        used = set()
        rows = []
        # We want to minimize the max entry.
        # Greedy: for each row, pick the smallest next number that doesn't create a duplicate difference.
        # But this might lead to a large scope if we're not careful.
        
        # A known technique for difference sets is to use "perfect rulers" or "sparse rulers".
        # Here we just need distinct differences.
        
        # Let's try to build rows that are "compact".
        # For a single row, the greedy approach (Golomb ruler) produces a ruler with length roughly k^2.
        # With multiple rows, we share the difference pool, so we can be denser? No, the differences must be distinct.
        # So the total number of differences is n_rows * k(k+1)/2.
        # The maximum difference is at most scope.
        # So scope must be at least n_rows * k(k+1)/2.
        # For n=7, k=6: 7 * 21 = 147 differences.
        # Max diff is scope. So scope >= 147? No, the differences are not all distinct values from 1 to scope.
        # They are just distinct.
        # The counting bound mentioned in the prompt is 105.
        # 7 * 21 = 147. This is the number of differences.
        # The largest difference is the scope.
        # The differences are integers from 1 to scope.
        # So we need 147 distinct integers in [1, scope].
        # Thus scope >= 147?
        # Wait, the prompt says "simple counting bound is 105".
        # Let's re-read: "all differences between two entries of the same row are distinct over all rows".
        # Number of differences per row: C(k+1, 2) = k(k+1)/2.
        # For k=6, C(7,2) = 21.
        # Total differences = 7 * 21 = 147.
        # If the largest entry is M, the differences are in [1, M].
        # So we need at least 147 distinct integers in [1, M].
        # So M >= 147.
        # Why does the prompt say 105?
        # Maybe k is the number of intervals? "k+1 integers". Yes.
        # Maybe the counting bound is different?
        # "simple counting bound is 105".
        # 105 is C(15,2)? No.
        # 105 is 7 * 15?
        # If k=5, C(6,2)=15. 7*15=105.
        # Ah, the task title is "Difference triangle set (7,5)".
        # But the function signature is construct(n_rows, k).
        # The prompt says "Choose 7 rows of 6 increasing integers".
        # So n_rows=7, k=5?
        # "Each row is a list of k + 1 integers".
        # If k=5, row has 6 integers.
        # So the call is likely construct(7, 5).
        # Then differences per row = C(6,2) = 15.
        # Total differences = 7 * 15 = 105.
        # So the counting bound is 105.
        # The best known is 112.
        
        # So we are minimizing the max entry M.
        # M must be >= 105.
        
        # Greedy approach:
        # We maintain a set of used differences.
        # For each row, we start with [0].
        # We want to add the next smallest number x such that all differences x - y for y in row are not in used.
        # This is the standard greedy for Golomb rulers.
        # Doing this for multiple rows sequentially might not be optimal.
        # We can try different orders or randomize the choice of x if there are ties (there are no ties in "smallest valid x", but we can randomize the search space).
        
        # Better: use backtracking?
        # Given the small size (7 rows, 6 entries), backtracking might be feasible if we have a good heuristic.
        # But 150s is a lot.
        
        # Let's try a randomized greedy with restarts.
        # We will try to build the set of rows.
        # We can also try to "repair" or "optimize" the solution.
        
        best_rows = None
        best_scope = float('inf')
        
        for attempt in range(1000):
            if time.time() > deadline:
                break
                
            used = set()
            rows = []
            success = True
            
            # Randomize the order of rows? Or just the choices?
            # The choices are deterministic in greedy.
            # We can add randomness by considering the top N smallest valid x and picking one at random.
            
            for r in range(n_rows):
                row = [0]
                # We want to fill the row with k more numbers.
                while len(row) < k + 1:
                    # Find the smallest x > row[-1] such that no difference conflicts.
                    # We can search upwards.
                    # To add randomness, we can look at the next few candidates.
                    candidates = []
                    x = row[-1] + 1
                    # Limit the search range for candidates to avoid infinite loops if stuck?
                    # If we go too high, we are failing.
                    # Let's search up to a reasonable limit, e.g., current best_scope or 200.
                    limit = best_scope if best_scope < float('inf') else 200
                    
                    found = False
                    while x <= limit:
                        new_diffs = {x - y for y in row}
                        if not (new_diffs & used):
                            candidates.append(x)
                            if len(candidates) >= 5:
                                break
                        x += 1
                    
                    if not candidates:
                        success = False
                        break
                    
                    # Pick randomly from candidates
                    x = rng.choice(candidates)
                    row.append(x)
                    used.update({x - y for y in row[:-1]})
                    
                if not success:
                    break
                
                rows.append(row)
                current_scope = max(row)
                if current_scope > best_scope:
                    success = False
                    break
            
            if success:
                scope = max(max(row) for row in rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [row[:] for row in rows]
                    if best_scope == 105: # Theoretical minimum
                        break
                        # Actually, 105 is the counting bound, but is it achievable?
                        # If yes, we can stop.
            
            # If we found a good solution, we can try to polish it?
            # Or just keep searching.
            
        if best_rows:
            return best_rows
        else:
            # Fallback: return a valid but possibly large scope solution
            # Greedy without randomness
            used = set()
            rows = []
            for r in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    new = {x - a for a in row}
                    if not new & used:
                        row.append(x)
                        used |= new
                rows.append(row)
            return rows

    return build_greedy(n_rows, k, rng)
# EVOLVE-BLOCK-END
