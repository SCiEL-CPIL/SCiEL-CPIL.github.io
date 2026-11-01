import time
import random
import math
import heapq

def construct(n_rows, k):
    start_time = time.time()
    deadline = 145.0  # 5s margin for safety

    # Total number of differences per row
    num_diffs_per_row = k * (k + 1) // 2
    
    # Lower bound on M:
    # We need n_rows * num_diffs_per_row distinct differences.
    # The largest difference is M.
    # Differences are in 1..M.
    # So n_rows * num_diffs_per_row <= M(M+1)/2 roughly? 
    # Actually, differences are just distinct integers.
    # The set of all differences is a subset of {1, ..., M}.
    # So M must be at least n_rows * num_diffs_per_row.
    # But also, the largest difference in a row is M.
    # Let's use the counting bound: M(M+1)/2 >= n_rows * num_diffs_per_row is not quite right 
    # because differences don't have to cover all 1..M, but they must be distinct.
    # The simplest bound: M >= n_rows * num_diffs_per_row is necessary if we assume max diff is M and we pack them tightly?
    # No, the differences are just distinct values. The max value is M.
    # So the set of all differences $D$ has size $N = n\_rows \times num\_diffs\_per\_row$.
    # All elements of $D$ are in $[1, M]$.
    # So $M \ge N$.
    
    N_total_diffs = n_rows * num_diffs_per_row
    min_M = N_total_diffs
    
    # A tighter bound:
    # For a single row with max M, the max diff is M.
    # If we have n_rows rows, the union of their diffs must have size N.
    # The largest possible diff is M.
    # So M must be at least N.
    
    # However, the structure of Sidon sets restricts how small M can be.
    # Let's start from min_M and go up.
    
    def generate_sidon_rows(max_M, count_limit=10000):
        """
        Generate valid Sidon rows (sets of k+1 ints starting with 0) with max entry <= max_M.
        Returns a list of sets of differences and the rows themselves.
        """
        rows = []
        # We will use a greedy/backtracking approach to generate rows
        # To speed up, we can randomize the selection of next element.
        
        def gen_row(idx):
            if idx == k:
                return [0] + rows
            # Try values from 1 to max_M
            # Optimization: start from small values to keep M low? 
            # But we want diverse rows.
            # Let's just pick random valid extensions.
            
            # Actually, generating many distinct Sidon rows is hard.
            # Let's use a randomized greedy construction for each row.
            pass
            
        # Simpler: Generate random rows and filter?
        # For small k (k=5), we can generate rows by picking random increasing sequences and checking validity.
        
        candidates = []
        seen_diff_sets = set() # To avoid duplicate rows? Not strictly necessary, but good for diversity.
        
        attempts = 0
        max_attempts = 50000
        while len(candidates) < count_limit and attempts < max_attempts and time.time() < deadline - 2.0:
            attempts += 1
            # Generate a random increasing sequence of length k+1 starting with 0, max <= max_M
            # We want to bias towards smaller max values?
            # Let's pick random positions.
            
            # To ensure max <= max_M, we pick k values from 1..max_M
            # But we want them to be "dense" enough to form a Sidon set?
            # Actually, random dense sets are often not Sidon.
            # Random sparse sets are more likely Sidon.
            
            # Strategy: Pick k random distinct integers from 1..max_M, sort them, prepend 0.
            # Check if it's Sidon.
            
            if max_M < k + 1:
                break
                
            vals = random.sample(range(1, max_M + 1), k)
            vals.sort()
            row = [0] + vals
            
            # Check Sidon property
            diffs = set()
            valid = True
            for i in range(k + 1):
                for j in range(i + 1, k + 1):
                    d = row[j] - row[i]
                    if d in diffs:
                        valid = False
                        break
                    diffs.add(d)
                if not valid:
                    break
            
            if valid:
                # Store the row and its diff set
                # We want to minimize the max entry, which is row[-1]
                candidates.append((row[-1], row, diffs))
                
        # Sort by max entry
        candidates.sort(key=lambda x: x[0])
        return candidates

    def find_solution_for_M(M):
        """
        Try to find n_rows non-conflicting rows with max entry <= M.
        Uses a conflict graph approach.
        """
        # 1. Generate candidates
        candidates = generate_sidon_rows(M, count_limit=2000)
        
        if len(candidates) < n_rows:
            return None

        # 2. Build conflict graph
        # Node: index in candidates
        # Edge: if diff sets intersect
        
        n_cand = len(candidates)
        
        # To save memory and time, we don't build the full graph explicitly if n_cand is large.
        # We will use a greedy selection with backtracking or local search.
        
        # Let's try a greedy selection first, prioritizing low max entry.
        # Candidates are already sorted by max entry.
        
        # We want to pick n_rows candidates such that no two share a diff.
        
        # Heuristic: 
        # Iterate through candidates.
        # If a candidate doesn't conflict with already picked ones, pick it.
        # This is greedy and might fail if we pick a "bad" low-M row early.
        
        # Better: Use a priority queue or local search.
        # Let's use a simple backtracking search with ordering.
        
        # Since n_rows is small (e.g., 7), we can do a DFS.
        # Order candidates by max entry (ascending).
        
        picked = []
        used_diffs = set()
        
        # To improve success rate, we can randomize the order of candidates with the same max entry,
        # or try multiple random permutations of the candidate list (filtered by max <= M).
        
        # Let's try multiple random restarts of the greedy selection.
        # 1. Filter candidates with max <= M.
        # 2. Shuffle them.
        # 3. Greedily pick non-conflicting ones.
        
        valid_candidates = [c for c in candidates if c[0] <= M]
        
        if len(valid_candidates) < n_rows:
            return None

        for _ in range(50): # 50 restarts
            random.shuffle(valid_candidates)
            
            curr_picked = []
            curr_used_diffs = set()
            
            for val, row, diffs in valid_candidates:
                # Check conflict
                if not curr_used_diffs.intersection(diffs):
                    curr_picked.append(row)
                    curr_used_diffs.update(diffs)
                    
                    if len(curr_picked) == n_rows:
                        return curr_picked
            
        return None

    # Search for minimal M
    best_solution = None
    
    # Start from a reasonable lower bound
    # The simple counting bound is M >= N_total_diffs.
    # But usually M is larger.
    # Let's start from min_M and increase.
    # To save time, we can jump.
    
    current_M = min_M
    
    # If min_M is too low, we might waste time generating candidates that are too sparse.
    # Let's estimate a better start.
    # For (7,5), N=105. Best known 112.
    # So start near 105.
    
    while time.time() < deadline:
        sol = find_solution_for_M(current_M)
        if sol:
            best_solution = sol
            break
        
        current_M += 1
        
        # If we are getting nowhere, maybe jump?
        # For now, linear increase is safe.
        
    if best_solution is None:
        # Fallback: Greedy construction from scratch (like original)
        # This ensures we return *something* valid if the graph approach fails.
        def greedy_init():
            used = set()
            rows = []
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    new_diffs = set()
                    conflict = False
                    for a in row:
                        d = x - a
                        if d in used:
                            conflict = True
                            break
                        new_diffs.add(d)
                    if conflict:
                        continue
                    row.append(x)
                    used.update(new_diffs)
                rows.append(row)
            return rows

        best_solution = greedy_init()
        
    return best_solution
