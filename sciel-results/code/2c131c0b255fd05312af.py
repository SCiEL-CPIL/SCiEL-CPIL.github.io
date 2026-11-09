import time
import random
import sys

def greedy_construct(n_rows, k):
    """
    Standard greedy construction to establish a baseline.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            diffs = {x - a for a in row}
            if not diffs & used:
                row.append(x)
                used.update(diffs)
        rows.append(row)
    return rows

def is_valid(rows, k):
    """Check if the set of rows is valid (all differences distinct)."""
    all_diffs = set()
    for r in rows:
        if len(r) != k + 1:
            return False
        if r[0] != 0:
            return False
        for i in range(1, len(r)):
            if r[i] <= r[i-1]:
                return False
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d = r[j] - r[i]
                if d in all_diffs:
                    return False
                all_diffs.add(d)
    return True

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # 1. Greedy Baseline
    best_rows = greedy_construct(n_rows, k)
    best_scope = max(max(r) for r in best_rows)
    
    # 2. Local Search (Simulated Annealing) over difference sets
    # State: list of sets of differences for each row
    # Cost: max(row_max)
    
    def get_row_max(diffs):
        """
        Given a set of differences, compute the maximum entry in the row.
        We want the smallest max entry such that all differences are present.
        This is equivalent to finding the smallest M such that the set of differences
        is a subset of the differences of a row of length k+1 with max M.
        Actually, we can just simulate the greedy construction of the row from the diffs,
        but we need to be careful: the diffs define the row uniquely if we start at 0?
        No, a set of differences does not uniquely define the row unless we know the order.
        However, for a "valid" row, the differences are distinct.
        If we have a set of differences D, we can try to reconstruct the row.
        The row is a set of integers {0, x1, ..., xk} with xk = M.
        The differences are {xj - xi}.
        If we just pick random subsets of differences, it's hard to ensure they form a valid row.
        
        Alternative Strategy:
        Represent the state as the rows themselves (lists of integers).
        Move: Pick a row, pick an element, change it to a new valid value.
        Constraint: All differences must be distinct across all rows.
        Cost: max(max(row) for row in rows).
        
        This is a huge space.
        
        Let's try a different local search:
        State: A set of differences assigned to each row.
        We need to ensure that the differences for each row can be realized by a strictly increasing sequence starting at 0.
        A set of differences D can be realized by a row iff there exists a set of integers {0, a1, ..., ak} such that {aj - ai | i<j} = D.
        This is a strong constraint.
        
        Let's stick to the row representation but use a smarter move.
        Move: 
        1. Pick a row r.
        2. Pick a position p in r.
        3. Try to change r[p] to a new value v such that:
           a. The row remains strictly increasing.
           b. The new differences introduced by r[p] do not conflict with existing used differences (excluding those from r[p] itself).
        
        This is essentially the same as the randomized greedy but with backtracking/mutation.
        """
        pass

    # Let's implement the Local Search over Rows directly.
    # Initial state: Greedy result.
    # Used differences: Set of all diffs.
    
    def calc_scope(rows):
        return max(max(r) for r in rows)

    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    current_rows = [list(r) for r in best_rows]
    current_scope = best_scope
    
    # Precompute used diffs
    def get_used_diffs(rows):
        used = set()
        for r in rows:
            used.update(get_diffs(r))
        return used

    current_used = get_used_diffs(current_rows)
    
    T = 50.0  # Initial temperature
    T_min = 0.1
    alpha = 0.999  # Cooling rate
    
    # We will run multiple restarts if time permits
    # But first, let's try to improve the greedy solution via SA.
    
    steps = 0
    while time.time() < deadline - 5:
        if T < T_min:
            # Restart with higher T
            T = 20.0
            # Maybe re-initialize to greedy? Or keep current?
            # Keeping current is better for local refinement, but greedy might be better for escape.
            # Let's keep current state.
            
        # Pick a row to mutate
        r_idx = random.randint(0, n_rows - 1)
        row = current_rows[r_idx]
        
        # Pick a position to mutate (not 0)
        p_idx = random.randint(1, len(row) - 1)
        old_val = row[p_idx]
        
        # Remove old diffs from used
        old_diffs = set()
        for i in range(len(row)):
            if i != p_idx:
                d = abs(row[p_idx] - row[i])
                old_diffs.add(d)
        
        for d in old_diffs:
            # Check if this diff is used by other rows
            # We need a count or a way to know if it's unique to this row.
            # Simpler: Recompute used diffs for the whole set? Too slow.
            # Let's maintain a count of how many times each diff is used.
            pass
        
        # The "used" set approach is tricky with mutations because we don't know if a diff is shared.
        # Let's use a different representation:
        # Just check validity of the move against the global set of diffs from OTHER rows.
        
        # Get diffs from all other rows
        other_diffs = set()
        for i in range(n_rows):
            if i != r_idx:
                other_diffs.update(get_diffs(current_rows[i]))
        
        # Try to find a new value for row[p_idx]
        # Range: [row[p_idx-1] + 1, row[p_idx+1] - 1] if p_idx < len-1 else [row[p_idx-1]+1, infinity]
        # To minimize scope, we prefer smaller values, but SA allows larger ones to escape.
        
        lower_bound = row[p_idx - 1] + 1
        if p_idx < len(row) - 1:
            upper_bound = row[p_idx + 1] - 1
        else:
            # Allow some expansion, but not too much
            upper_bound = current_scope + 10 
        
        if lower_bound > upper_bound:
            T *= alpha
            continue
            
        # Sample a new value
        # Bias towards smaller values to minimize scope, but allow random jumps
        # Use a geometric distribution or just uniform with bias?
        # Let's pick from a window around the current value, biased towards lower.
        window_size = 5 + int(T)
        start_val = max(lower_bound, old_val - window_size)
        end_val = min(upper_bound, old_val + window_size)
        
        if start_val > end_val:
            T *= alpha
            continue
            
        new_val = random.randint(start_val, end_val)
        
        if new_val == old_val:
            T *= alpha
            continue
            
        # Check if new_val creates conflicts
        # New diffs introduced: {new_val - row[i] for i != p_idx}
        # These must not be in other_diffs.
        # Also, the row must remain strictly increasing (ensured by bounds).
        
        new_diffs = set()
        conflict = False
        for i in range(len(row)):
            if i != p_idx:
                d = abs(new_val - row[i])
                if d in other_diffs:
                    conflict = True
                    break
                new_diffs.add(d)
        
        if conflict:
            T *= alpha
            continue
            
        # Calculate new scope
        new_row = row.copy()
        new_row[p_idx] = new_val
        # Ensure strict increase (already checked by bounds, but let's be safe)
        is_increasing = True
        for i in range(1, len(new_row)):
            if new_row[i] <= new_row[i-1]:
                is_increasing = False
                break
        if not is_increasing:
            T *= alpha
            continue
            
        new_scope = max(max(r) for r in current_rows) # This is inefficient, we should track max
        # Actually, max_scope = max(max(current_rows[i]) for i in range(n_rows) if i != r_idx, max(new_row))
        other_max = max(max(current_rows[i]) for i in range(n_rows) if i != r_idx)
        new_row_max = max(new_row)
        new_scope_val = max(other_max, new_row_max)
        
        old_scope_val = current_scope
        
        delta = new_scope_val - old_scope_val
        
        # Acceptance criterion
        if delta < 0 or random.random() < pow(2.71828, -delta / T):
            # Accept
            current_rows[r_idx] = new_row
            current_scope = new_scope_val
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in current_rows]
        
        T *= alpha
        steps += 1

    # Final verification
    if is_valid(best_rows, k):
        return best_rows
    else:
        # Fallback to greedy if something went wrong
        return greedy_construct(n_rows, k)

# To satisfy the interface and provide a robust solution, we wrap the SA in a construct function
# The above construct is the main logic.

# Let's refine the SA to be faster by precomputing other_diffs less often or using a smarter data structure.
# But for n_rows=4, k=7, the size is small (8 elements per row).
# other_diffs has size 3 * 28 = 84 elements. Checking conflict is O(28) per move.
# This is very fast.

# One issue: The initial greedy might be far from the optimum.
# We can try multiple random restarts of the SA from different initial states.
# But the greedy is a good starting point.

# Let's add a "randomized greedy" initialization for restarts.

def randomized_greedy_init(n_rows, k, seed):
    random.seed(seed)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            candidates = []
            # Look for valid next marks
            # To keep it fast, we don't look too far ahead in this init
            for x in range(current_max + 1, current_max + 50):
                diffs = {x - a for a in row}
                if not (diffs & used):
                    candidates.append(x)
            if not candidates:
                return None
            # Pick randomly
            next_x = random.choice(candidates)
            row.append(next_x)
            current_max = next_x
            used.update({next_x - a for a in row[:-1]})
    return rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140
    
    # 1. Greedy Baseline
    best_rows = greedy_construct(n_rows, k)
    best_scope = max(max(r) for r in best_rows)
    
    # 2. Local Search
    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def run_sa(initial_rows, start_time, deadline):
        current_rows = [list(r) for r in initial_rows]
        current_scope = max(max(r) for r in current_rows)
        
        T = 10.0
        T_min = 0.1
        alpha = 0.9995
        
        while time.time() < deadline - 2:
            if T < T_min:
                T = 5.0
                
            r_idx = random.randint(0, n_rows - 1)
            row = current_rows[r_idx]
            p_idx = random.randint(1, len(row) - 1)
            old_val = row[p_idx]
            
            other_diffs = set()
            for i in range(n_rows):
                if i != r_idx:
                    other_diffs.update(get_diffs(current_rows[i]))
            
            lower_bound = row[p_idx - 1] + 1
            if p_idx < len(row) - 1:
                upper_bound = row[p_idx + 1] - 1
            else:
                upper_bound = current_scope + 20
                
            if lower_bound > upper_bound:
                T *= alpha
                continue
                
            window_size = 5 + int(T)
            start_val = max(lower_bound, old_val - window_size)
            end_val = min(upper_bound, old_val + window_size)
            
            if start_val > end_val:
                T *= alpha
                continue
                
            new_val = random.randint(start_val, end_val)
            
            if new_val == old_val:
                T *= alpha
                continue
                
            conflict = False
            for i in range(len(row)):
                if i != p_idx:
                    d = abs(new_val - row[i])
                    if d in other_diffs:
                        conflict = True
                        break
            
            if conflict:
                T *= alpha
                continue
                
            new_row = row.copy()
            new_row[p_idx] = new_val
            
            is_increasing = True
            for i in range(1, len(new_row)):
                if new_row[i] <= new_row[i-1]:
                    is_increasing = False
                    break
            if not is_increasing:
                T *= alpha
                continue
                
            other_max = max(max(current_rows[i]) for i in range(n_rows) if i != r_idx)
            new_row_max = max(new_row)
            new_scope_val = max(other_max, new_row_max)
            
            old_scope_val = current_scope
            delta = new_scope_val - old_scope_val
            
            if delta < 0 or random.random() < pow(2.71828, -delta / T):
                current_rows[r_idx] = new_row
                current_scope = new_scope_val
                
            T *= alpha
            
        return current_rows, current_scope

    # Run SA from Greedy
    rows_sa, scope_sa = run_sa(best_rows, start_time, deadline)
    if scope_sa < best_scope and is_valid(rows_sa, k):
        best_scope = scope_sa
        best_rows = rows_sa
        
    # Try a few random restarts if time permits
    if time.time() < deadline - 30:
        for seed in range(10):
            if time.time() > deadline - 10:
                break
            init_rows = randomized_greedy_init(n_rows, k, seed)
            if init_rows is None:
                continue
            init_scope = max(max(r) for r in init_rows)
            if init_scope < best_scope:
                best_scope = init_scope
                best_rows = init_rows
                
            # Run SA on this init
            rows_sa, scope_sa = run_sa(init_rows, start_time, deadline)
            if scope_sa < best_scope and is_valid(rows_sa, k):
                best_scope = scope_sa
                best_rows = rows_sa
                
    return best_rows
