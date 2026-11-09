import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    n = 192
    max_size = 48
    
    # Helper to check if a set is valid
    def is_valid(S):
        S_list = list(S)
        k = len(S_list)
        if k < 4:
            return True
        
        # Precompute all sums of 3 distinct elements
        # We need to check for each a in S, if a divides any sum of 3 others
        # Optimization: Group by a?
        
        # For each a in S:
        #   Exclude a from the pool
        #   Check if a divides sum(b,c,d) for any b,c,d in S\{a}
        
        # Since k is small (~48), direct iteration is fine.
        # But we can be smarter.
        
        for a in S_list:
            others = [x for x in S_list if x != a]
            # We need to check if a divides sum of any 3 in others
            # If a is large, few multiples.
            # If a is small, many.
            
            # Iterate combinations of 3 from others
            # To speed up, we can break early if found.
            for b, c, d in combinations(others, 3):
                if (b + c + d) % a == 0:
                    return False
        return True

    def get_violations(S):
        """Returns a list of (a, b, c, d) where a | (b+c+d) and a,b,c,d in S distinct."""
        S_list = list(S)
        violations = []
        for a in S_list:
            others = [x for x in S_list if x != a]
            for b, c, d in combinations(others, 3):
                if (b + c + d) % a == 0:
                    violations.append((a, b, c, d))
        return violations

    def count_violations_per_element(S):
        """Counts how many times each element is involved in a violation."""
        S_list = list(S)
        counts = {x: 0 for x in S_list}
        violations = get_violations(S)
        for a, b, c, d in violations:
            counts[a] += 1
            counts[b] += 1
            counts[c] += 1
            counts[d] += 1
        return counts

    # Strategy 1: Greedy Removal from Top Pool
    # Start with the largest 48 numbers
    # If they are valid, we are done.
    # If not, remove the most problematic ones.
    
    initial_pool = list(range(n - max_size + 1, n + 1)) # 145 to 192
    S = set(initial_pool)
    
    # Check validity
    if is_valid(S):
        return list(S)
    
    # Repair loop
    while not is_valid(S) and time.time() < deadline:
        counts = count_violations_per_element(S)
        if not any(v > 0 for v in counts.values()):
            break # Should not happen if is_valid is False
        
        # Find element with max violations
        max_count = max(counts.values())
        # Remove one of the elements with max count
        # Prefer removing larger elements? Or smaller?
        # Usually, removing the element that acts as a divisor often breaks many chains.
        # But removing a summand also helps.
        # Let's remove the one with the highest count. Tie-break by value (remove larger to keep density high? Or smaller to keep range wide?)
        # Let's try removing the one with the highest count, tie-break by largest value.
        candidates = [x for x in S if counts[x] == max_count]
        x_to_remove = max(candidates)
        
        S.remove(x_to_remove)
        
        # If we dropped below max_size, try to add back some
        if len(S) < max_size:
            # Try to add from the rest of the universe
            # Prioritize numbers close to the current set
            remaining = [x for x in range(1, n + 1) if x not in S]
            # Sort remaining by how "compatible" they are? 
            # Just try adding in descending order first
            for x in sorted(remaining, reverse=True):
                if time.time() > deadline:
                    break
                S_trial = S | {x}
                if is_valid(S_trial):
                    S = S_trial
                    if len(S) >= max_size:
                        break

    # If we have a valid set of size < max_size, try to improve
    if len(S) < max_size:
        # Try local search: swap elements
        # For each x in S, for each y not in S, check if (S - {x}) U {y} is valid
        # This is expensive, so do it only if we have time
        improved = True
        while improved and time.time() < deadline:
            improved = False
            S_list = list(S)
            remaining = [x for x in range(1, n + 1) if x not in S]
            for x in S_list:
                if time.time() > deadline:
                    break
                S_minus_x = S - {x}
                # Try to find a y that fits
                # Sort remaining by heuristic? e.g., large numbers first
                for y in sorted(remaining, reverse=True):
                    S_trial = S_minus_x | {y}
                    if is_valid(S_trial):
                        S = S_trial
                        improved = True
                        break # Restart loop with new S
                if improved:
                    break

    return list(S)
