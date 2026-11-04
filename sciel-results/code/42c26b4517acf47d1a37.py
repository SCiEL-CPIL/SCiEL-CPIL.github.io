import time
import random
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    
    Strategy: Simulated Annealing over partitions of differences.
    1. Determine the set of required differences for a given scope M.
    2. Randomly assign these differences to n_rows "bins" (rows).
    3. For each bin, try to reconstruct a valid row of k+1 elements 
       (starting at 0) that generates exactly those differences.
    4. Mutate the assignment by swapping differences between bins.
    5. Accept mutations based on a cost function (number of invalid bins) 
       and a temperature schedule.
    """
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill

    num_elements = k + 1
    total_diffs_per_row = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * total_diffs_per_row

    # Helper: Check if a set of differences can form a valid row starting at 0
    # Returns the row if valid, else None
    def reconstruct_row(diffs, max_scope):
        """
        Given a set of differences, try to find a row [0, x1, ..., x_{k}]
        such that all pairwise differences are in 'diffs' and vice versa.
        This is a small search problem.
        """
        if len(diffs) != total_diffs_per_row:
            return None
        
        # The largest difference in the row must be the max value of the row.
        # Let M_row = max(diffs). The row must end at M_row?
        # Not necessarily. If the row is [0, a, b, c], diffs are a, b, c, b-a, c-a, c-b.
        # max(diff) = c. So the last element is max(diff).
        # Wait, if the row is [0, 2, 5], diffs: 2, 5, 3. Max is 5. Last element is 5.
        # So the last element of the row is always max(diffs).
        
        max_val = max(diffs)
        if max_val > max_scope:
            return None
            
        # We need to find k-1 intermediate values 0 < x1 < ... < x_{k-1} < max_val
        # such that the set of differences matches 'diffs'.
        
        # Heuristic search for the intermediate points.
        # We know 0 and max_val are in the row.
        # The remaining k-1 points must be chosen from {1, ..., max_val-1}.
        
        # To make this fast, we can use backtracking with strong pruning.
        # The set of diffs is fixed. 
        # If we pick a point x, it must create diffs with 0 (which is x) and max_val (which is max_val-x).
        # So x and max_val-x must be in 'diffs'.
        
        candidates = []
        for d in diffs:
            if d < max_val:
                # If we pick x=d, then max_val-d must also be in diffs.
                if max_val - d in diffs:
                    candidates.append(d)
        
        # We need to select k-1 distinct candidates from this list such that 
        # all pairwise differences among them are also in 'diffs'.
        
        # Sort candidates to help with ordering? The row must be increasing.
        # Let's just use a simple backtracking search over the candidates.
        
        def backtrack(pos, current_points, used_local_diffs):
            if pos == k - 1:
                # Check if all pairwise diffs of current_points are in diffs
                # And if the union of diffs from current_points + {0, max_val} is exactly 'diffs'
                # Actually, we just need to ensure no conflicts and full coverage.
                
                # Reconstruct full row
                full_row = [0] + sorted(current_points) + [max_val]
                
                # Verify
                generated_diffs = set()
                for i in range(len(full_row)):
                    for j in range(i+1, len(full_row)):
                        d = full_row[j] - full_row[i]
                        if d not in diffs:
                            return None
                        generated_diffs.add(d)
                
                if generated_diffs == diffs:
                    return full_row
                return None
            
            for c in candidates:
                if c in current_points:
                    continue
                
                # Pruning: Check if c creates valid diffs with existing points
                valid = True
                for p in current_points:
                    d1 = abs(c - p)
                    if d1 not in diffs:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Also check if c and max_val-c are consistent? 
                # The condition "c and max_val-c in diffs" was used to generate candidates.
                # But we must ensure we don't pick both c and max_val-c if they conflict?
                # No, they can both be in the row if they are distinct points.
                
                # Add c
                current_points.append(c)
                res = backtrack(pos + 1, current_points, used_local_diffs)
                if res:
                    return res
                current_points.pop()
            
            return None
        
        # Optimization: The candidates list might be large. 
        # We should prioritize candidates that are "central" or have many connections?
        # For now, randomize order to help with diverse restarts in SA.
        random.shuffle(candidates)
        
        return backtrack(0, [], set())

    def evaluate_partition(assignment, max_scope):
        """
        assignment: list of sets, one for each row.
        Returns: (cost, rows)
        cost: number of rows that are invalid.
        rows: list of reconstructed rows (None if invalid).
        """
        rows = []
        cost = 0
        for i in range(n_rows):
            diffs = assignment[i]
            row = reconstruct_row(diffs, max_scope)
            if row is None:
                cost += 1
                rows.append(None)
            else:
                # Verify row is strictly increasing and starts at 0 (reconstruct_row ensures this)
                rows.append(row)
        
        return cost, rows

    def solve_with_sa(M):
        """
        Try to find a valid configuration with scope M using Simulated Annealing.
        """
        # 1. Generate the pool of differences
        # We need exactly total_diffs_needed differences.
        # To minimize scope M, we should use the smallest possible differences?
        # Not necessarily. The differences must fit into the rows.
        # However, the set of all differences used in the solution is a subset of {1, ..., M}.
        # Let's start with the smallest total_diffs_needed differences: 1 to total_diffs_needed.
        # If M is small, this might be the only option.
        # If M is large, we can pick any subset.
        
        # Actually, the differences in the rows are determined by the rows.
        # The union of all differences must be a set of size total_diffs_needed.
        # Let's assume the differences are exactly {1, 2, ..., total_diffs_needed}.
        # Why? Because if we use a larger difference, we waste "space" in the difference domain.
        # Wait, the differences are values in the integer line. 
        # If we use difference 100, it doesn't prevent using difference 1.
        # The constraint is that the *rows* must fit in [0, M].
        # The differences themselves don't have a global upper bound other than M.
        # But to minimize M, we generally want small differences.
        
        # Let's try to find a solution where the set of used differences is exactly {1, ..., total_diffs_needed}.
        # This is a strong constraint but might be valid for optimal solutions.
        
        diff_pool = list(range(1, total_diffs_needed + 1))
        
        # Initial random assignment
        assignment = [set() for _ in range(n_rows)]
        random.shuffle(diff_pool)
        for i, d in enumerate(diff_pool):
            assignment[i % n_rows].add(d)
            
        # Ensure each bin has the correct number of diffs
        # The above round-robin might not give equal counts if not divisible, but it is.
        # 7 * 21 = 147. 147 % 7 == 0.
        
        cost, rows = evaluate_partition(assignment, M)
        
        if cost == 0:
            return rows
            
        # Simulated Annealing
        T = 1.0
        cooling = 0.999
        min_T = 0.001
        
        current_cost = cost
        current_assignment = [set(s) for s in assignment]
        
        while T > min_T:
            if time.time() > DEADLINE:
                return None
                
            # Mutation: Pick two random rows i, j and two random diffs d_i, d_j and swap them
            i, j = random.sample(range(n_rows), 2)
            if not current_assignment[i] or not current_assignment[j]:
                continue
                
            d_i = random.choice(list(current_assignment[i]))
            d_j = random.choice(list(current_assignment[j]))
            
            if d_i == d_j:
                continue
                
            # Swap
            current_assignment[i].remove(d_i)
            current_assignment[i].add(d_j)
            current_assignment[j].remove(d_j)
            current_assignment[j].add(d_i)
            
            new_cost, _ = evaluate_partition(current_assignment, M)
            
            delta = new_cost - current_cost
            if delta <= 0 or random.random() < pow(2.718, -delta / T):
                current_cost = new_cost
                if current_cost == 0:
                    # Found a valid partition!
                    _, rows = evaluate_partition(current_assignment, M)
                    return rows
            else:
                # Revert
                current_assignment[i].remove(d_j)
                current_assignment[i].add(d_i)
                current_assignment[j].remove(d_i)
                current_assignment[j].add(d_j)
                
            T *= cooling
            
        return None

    # Try scopes from 105 upwards
    # We might need to randomize the diff pool selection if {1..N} doesn't work.
    # But let's start with the tightest packing.
    
    best_rows = None
    
    # Try a few random seeds for the SA to increase chances
    for M in range(105, 113):
        if time.time() > DEADLINE:
            break
            
        # Try multiple restarts for each M
        for restart in range(5):
            if time.time() > DEADLINE:
                break
                
            random.seed(42 + M * 100 + restart) # Deterministic restarts for reproducibility in testing, but random in prod?
            # Actually, using random.seed makes it deterministic. 
            # For the evaluator, we just need to be fast. 
            # Let's not seed, or seed based on time? 
            # The prompt says "Python standard library...". 
            # Using random without seed is fine.
            
            res = solve_with_sa(M)
            if res is not None:
                # Verify validity
                all_diffs = []
                for r in res:
                    for i in range(len(r)):
                        for j in range(i+1, len(r)):
                            all_diffs.append(r[j] - r[i])
                if len(all_diffs) == len(set(all_diffs)):
                    max_entry = max(max(r) for r in res)
                    if max_entry <= M:
                        return res
                    else:
                        # Should not happen if reconstruct_row respects max_scope
                        return res

    # Fallback: Greedy construction (valid but potentially larger scope)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < num_elements:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    
    return rows

if __name__ == "__main__":
    # Quick test
    res = construct(7, 6)
    print(f"Scope: {max(max(r) for r in res)}")
    all_diffs = []
    for r in res:
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                all_diffs.append(r[j] - r[i])
    print(f"Valid: {len(all_diffs) == len(set(all_diffs))}")
    print(res)
