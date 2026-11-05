"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    deadline = time.time() + 145.0

    def all_diffs(rows):
        diffs = []
        for row in rows:
            n = len(row)
            for i in range(n):
                for j in range(i + 1, n):
                    diffs.append(row[j] - row[i])
        return diffs

    def max_entry(rows):
        return max(max(row) for row in rows)

    def is_valid(rows):
        diffs = all_diffs(rows)
        return len(diffs) == len(set(diffs))

    # --- Greedy initialization ---
    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
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

    best_rows = greedy_init()
    best_max = max_entry(best_rows)

    # --- Conflict graph formulation ---
    # For a given scope M, build conflict graph where nodes are candidate row entries
    # and edges represent difference collisions. Then find independent set.

    def compute_conflict_graph(M):
        """
        For scope M, each row has k+1 entries (positions 0..k).
        Position 0 is always 0. Positions 1..k have values in 1..M.
        
        We model this as: for each row r and position p (1..k), the value v must satisfy:
        - v > row[r][p-1] (strictly increasing within row)
        - All differences v - row[r][i] for i < p must not conflict with other rows
        - All differences row[r][j] - v for j > p must not conflict
        - Differences within same row must be distinct
        - Differences across rows must be distinct
        
        This is complex. Instead, we use a different approach:
        Build a candidate set of possible (row, pos, value) triples, then use
        maximum independent set on the conflict graph.
        
        But that's too large. Instead, we use constraint propagation + backtracking.
        """
        pass

    # --- Better approach: Use exact cover / DLX-like backtracking with pruning ---
    # For a fixed scope M, we try to assign values to each (row, pos) such that
    # all 112 differences are distinct and in range 1..M.
    
    def solve_for_M(M, time_limit=5.0):
        """
        Try to find a valid configuration with all entries <= M.
        Uses backtracking with constraint propagation.
        """
        start_time = time.time()
        
        # Total number of differences per row: k*(k+1)/2
        # For k=8: 36 differences per row
        # For n_rows=4: 144 differences total
        total_diffs = n_rows * k * (k + 1) // 2
        
        # We need total_diffs distinct differences, each in [1, M]
        # So we need M >= total_diffs
        if M < total_diffs:
            return None
        
        # Backtracking search
        # State: rows[r][p] for r in 0..n_rows-1, p in 0..k
        # rows[r][0] = 0 always
        rows = [[0] * (k + 1) for _ in range(n_rows)]
        used_diffs = set()
        
        # Order of assignment: row by row, position by position
        # For each (r, p), try values from lower+1 to min(M, upper-1)
        # where lower = rows[r][p-1], upper = rows[r][p+1] if p < k else M+1
        
        # For efficiency, we process in order and maintain used_diffs incrementally
        # When assigning rows[r][p] = v, the new differences are:
        #   v - rows[r][i] for i in 0..p-1
        #   rows[r][j] - v for j in p+1..k (but these are unknown yet)
        # So we can only check the first set immediately.
        
        # Actually, let's do it differently: assign all entries of one row at a time,
        # then check cross-row conflicts.
        
        def backtrack_row(r):
            if r == n_rows:
                return True
            if time.time() - start_time > time_limit:
                return False
            
            # Assign row r
            if not assign_row(r, 1, time_limit, start_time):
                return False
            
            return backtrack_row(r + 1)
        
        def assign_row(r, p, time_limit, start_time):
            if p > k:
                # Row r is complete, check if all differences are valid
                # Within-row differences
                for i in range(k + 1):
                    for j in range(i + 1, k + 1):
                        d = rows[r][j] - rows[r][i]
                        if d in used_diffs:
                            # Conflict with previous rows
                            return False
                        used_diffs.add(d)
                
                # Note: we also need to ensure within-row differences are distinct
                # Let's check that
                row_diffs = []
                for i in range(k + 1):
                    for j in range(i + 1, k + 1):
                        row_diffs.append(rows[r][j] - rows[r][i])
                if len(row_diffs) != len(set(row_diffs)):
                    return False
                
                return True
            
            if time.time() - start_time > time_limit:
                return False
            
            lower = rows[r][p - 1] + 1
            upper = M
            
            for v in range(lower, upper + 1):
                # Check if v - rows[r][i] conflicts with used_diffs for i < p
                conflict = False
                for i in range(p):
                    d = v - rows[r][i]
                    if d in used_diffs:
                        conflict = True
                        break
                    # Also check if this difference already used within this row
                    # (will be checked when row is complete)
                
                if conflict:
                    continue
                
                rows[r][p] = v
                
                # Recurse
                if assign_row(r, p + 1, time_limit, start_time):
                    return True
                
                rows[r][p] = 0
            
            return False
        
        # Initialize used_diffs as empty
        # But we need to handle the fact that we're building rows one by one
        # and the used_diffs set accumulates across rows
        
        # Reset
        used_diffs = set()
        rows = [[0] * (k + 1) for _ in range(n_rows)]
        
        if backtrack_row(0):
            return [row[:] for row in rows]
        
        return None

    # --- Multi-stage search ---
    # Stage 1: Try to improve from greedy solution using simulated annealing
    current_rows = [r[:] for r in best_rows]
    current_max = best_max

    indices = [(r, i) for r in range(n_rows) for i in range(1, k + 1)]

    T = 50.0
    T_min = 0.01
    cooling = 0.99995

    def compute_cost(rows):
        diffs = all_diffs(rows)
        counts = {}
        for d in diffs:
            counts[d] = counts.get(d, 0) + 1
        violations = sum(c - 1 for c in counts.values() if c > 1)
        m = max_entry(rows)
        return violations * 10000 + m

    current_cost = compute_cost(current_rows)

    while time.time() < deadline - 30.0:
        if T < T_min:
            break

        r, i = random.choice(indices)
        old_val = current_rows[r][i]

        lower = current_rows[r][i - 1] + 1
        upper = current_rows[r][i + 1] - 1 if i < k else current_rows[r][k] + 50

        if random.random() < 0.3:
            span = upper - lower + 1
            new_val = lower + int(random.random() * span * 0.5)
        else:
            new_val = random.randint(lower, upper)

        if new_val == old_val:
            continue

        current_rows[r][i] = new_val

        row = current_rows[r]
        if any(row[j] <= row[j - 1] for j in range(1, len(row))):
            current_rows[r][i] = old_val
            continue

        new_cost = compute_cost(current_rows)

        delta = new_cost - current_cost
        if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
            current_cost = new_cost
            new_max = max_entry(current_rows)
            if new_max < best_max:
                best_max = new_max
                best_rows = [rr[:] for rr in current_rows]
        else:
            current_rows[r][i] = old_val

        T *= cooling

    # Stage 2: Try exact backtracking for M values below current best
    if time.time() < deadline - 5.0:
        # Try a few M values from best_max - 1 down to a reasonable limit
        min_M = max(n_rows * k * (k + 1) // 2, best_max - 20)
        
        for M in range(best_max - 1, min_M - 1, -1):
            if time.time() > deadline - 5.0:
                break
            
            sol = solve_for_M(M, time_limit=2.0)
            if sol is not None:
                # Verify
                diffs = all_diffs(sol)
                if len(diffs) == len(set(diffs)):
                    best_rows = sol
                    best_max = M
                    break

    return best_rows
# EVOLVE-BLOCK-END
