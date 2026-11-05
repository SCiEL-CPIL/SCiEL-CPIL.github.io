"""Difference triangle set construction via randomized local search."""
import random
import time

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimize the largest entry.
    """
    # Total number of differences needed: n_rows * (k+1)*k/2
    total_diffs = n_rows * (k + 1) * k // 2
    
    # Simple counting bound: need at least total_diffs distinct positive integers
    # up to some max value M. The number of distinct differences <= M is M*(M+1)/2
    # So we need M*(M+1)/2 >= total_diffs
    import math
    # Solve M^2 + M - 2*total_diffs >= 0
    M_min = int(math.ceil((-1 + math.sqrt(1 + 8 * total_diffs)) / 2))
    
    # For (7,5): total_diffs = 7*15 = 105, M_min = ceil((-1+sqrt(1+840))/2) = ceil((-1+29.0)/2) = 14
    # But that's way too low because differences must be within rows starting at 0.
    # Actually the counting bound in the problem says 105, which is the max entry lower bound.
    # The real constraint is more complex.
    
    # Let's use a smarter approach: randomized greedy with backtracking/local search.
    
    best_rows = None
    best_max = float('inf')
    
    # Time limit
    start_time = time.time()
    time_limit = 110  # leave some margin
    
    # Strategy: Use a randomized approach where we try to build rows with smaller max values.
    # Start from a reasonable lower bound and try to find valid configurations.
    
    # Lower bound: the largest entry must be at least such that we can fit all differences.
    # For each row, the max entry m_i gives at most m_i distinct differences (actually more, but bounded).
    # A better lower bound: sum of max entries must accommodate all differences.
    
    # Let's try multiple random seeds and use a hybrid greedy + local search approach.
    
    def try_construct(max_val, seed):
        """Try to construct rows with all entries <= max_val."""
        rng = random.Random(seed)
        
        # Greedy with randomization: for each row, pick positions randomly but ensure validity
        used_diffs = set()
        rows = []
        
        for row_idx in range(n_rows):
            # Start with [0]
            row = [0]
            # We need k more elements, each between 1 and max_val
            # Use a randomized backtracking approach
            
            def backtrack(pos, current_row, current_diffs):
                if len(current_row) == k + 1:
                    return True
                if time.time() - start_time > time_limit:
                    return False
                
                # Try values from current_row[-1] + 1 to max_val
                # Randomize the order
                candidates = list(range(current_row[-1] + 1, max_val + 1))
                rng.shuffle(candidates)
                
                for x in candidates:
                    new_diffs = {x - a for a in current_row}
                    if not (new_diffs & used_diffs) and not (new_diffs & current_diffs):
                        # Check within-row uniqueness too (differences within same row must be distinct)
                        # Actually, differences within the same row are automatically distinct if elements are distinct and increasing?
                        # No! e.g., [0, 2, 4] has differences 2, 4, 2 - wait, 4-0=4, 4-2=2, 2-0=2. So 2 appears twice.
                        # We need all pairwise differences to be distinct within a row AND across rows.
                        
                        # Check if new_diffs has duplicates within itself
                        if len(new_diffs) != len(current_row):
                            continue
                            
                        # Also check that new_diffs doesn't conflict with existing current_diffs
                        if new_diffs & current_diffs:
                            continue
                            
                        # Also check new_diffs doesn't conflict with used_diffs (from previous rows)
                        if new_diffs & used_diffs:
                            continue
                        
                        # Additionally, we need to ensure that when we add x, the differences between x and all previous elements
                        # are all distinct from each other (already checked via len) and from all previous differences in this row
                        # (already checked via new_diffs & current_diffs)
                        
                        # But we also need to make sure that the differences among the existing elements are still valid
                        # They are, since we're just adding a new element.
                        
                        current_row.append(x)
                        current_diffs |= new_diffs
                        
                        if backtrack(pos + 1, current_row, current_diffs):
                            return True
                        
                        current_row.pop()
                        current_diffs -= new_diffs
            
        # This backtracking is too slow for large max_val. Let's use a different approach.
        # Instead, let's use a greedy approach with randomization and restarts.
        
        return None
    
    # Better approach: Use a simulated annealing / local search on the set of rows.
    # Start with a valid configuration (from greedy) and try to reduce the max value.
    
    # First, get a valid configuration using the original greedy method
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used) and len(new) == len(row):
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    initial_rows = greedy_construct()
    best_rows = initial_rows
    best_max = max(max(r) for r in initial_rows)
    
    # Now try to improve via local search
    # Represent the solution as a flat list of all row elements (excluding the leading 0s)
    # Or better, keep rows as lists and mutate them.
    
    def is_valid(rows):
        """Check if all differences are distinct across all rows."""
        used = set()
        for row in rows:
            row_diffs = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in row_diffs:
                        return False
                    if d in used:
                        return False
            row_diffs.update(row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row)))
            used |= row_diffs
        return True
    
    def get_max_val(rows):
        return max(max(r) for r in rows)
    
    # Local search: try to reduce the maximum entry by modifying rows
    # Mutation: pick a random row and try to replace one of its elements with a smaller value
    
    def mutate(rows, rng):
        """Create a mutated version of rows."""
        new_rows = [list(r) for r in rows]
        row_idx = rng.randint(0, n_rows - 1)
        # Pick a random position (not the first, which is 0)
        pos = rng.randint(1, k)
        old_val = new_rows[row_idx][pos]
        
        # Try to replace with a smaller value
        # The new value must be greater than the previous element and less than the next
        min_val = new_rows[row_idx][pos - 1] + 1
        max_val = new_rows[row_idx][pos + 1] - 1 if pos < k else old_val - 1
        
        if min_val > max_val:
            return None
        
        # Try a few random values in this range
        for _ in range(10):
            new_val = rng.randint(min_val, max_val)
            if new_val == old_val:
                continue
            candidate = list(new_rows[row_idx])
            candidate[pos] = new_val
            new_rows[row_idx] = candidate
            
            # Check if this row is still internally valid
            row_diffs = set()
            valid = True
            for i in range(k + 1):
                for j in range(i + 1, k + 1):
                    d = candidate[j] - candidate[i]
                    if d in row_diffs:
                        valid = False
                        break
                    row_diffs.add(d)
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check against other rows
            used = set()
            for ri, row in enumerate(new_rows):
                if ri == row_idx:
                    continue
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        used.add(row[j] - row[i])
            
            if row_diffs & used:
                continue
            
            return new_rows
        
        return None
    
    # Simulated annealing
    current_rows = initial_rows
    current_max = best_max
    temperature = 10.0
    cooling_rate = 0.99
    
    while time.time() - start_time < time_limit:
        if temperature < 0.1:
            break
            
        new_rows = mutate(current_rows, random)
        if new_rows is None:
            temperature *= cooling_rate
            continue
        
        new_max = get_max_val(new_rows)
        delta = new_max - current_max
        
        if delta <= 0 or random.random() < math.exp(-delta / temperature):
            current_rows = new_rows
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_rows = [list(r) for r in current_rows]
        
        temperature *= cooling_rate
    
    return best_rows
