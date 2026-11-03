import time
import random
import sys
import math
from itertools import combinations

def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]

    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    m = k + 1  # entries per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row

    def is_valid(rows):
        if len(rows) != n_rows:
            return False
        seen = set()
        for row in rows:
            if len(row) != m or row[0] != 0:
                return False
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    def row_diffs(row):
        s = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                s.add(row[j] - row[i])
        return s

    def scope_of(rows):
        return max(max(r) for r in rows)

    # --- Helper: Generate all valid rows for a given max scope ---
    def generate_valid_rows(max_scope):
        valid_rows = []
        # We need to generate all increasing sequences of length m starting at 0
        # such that max element <= max_scope and internal diffs are distinct.
        
        # To speed this up, we can use backtracking to build rows.
        # However, for small k and moderate max_scope, simple recursion is fine.
        
        def backtrack(pos, current_row, current_diffs):
            if time.time() > deadline - 5:
                return
            if pos == m:
                if max(current_row) <= max_scope:
                    # Check if this row is valid (it is by construction of current_diffs)
                    # But we must ensure no duplicate diffs within the row
                    # The backtrack logic ensures new diffs don't intersect current_diffs
                    valid_rows.append(list(current_row))
                return
            
            # Pruning: if current_row is empty, start from 1
            # If not, start from last + 1
            start_val = current_row[-1] + 1 if current_row else 1
            
            # Upper bound for next value
            # We need to fit (m - pos - 1) more elements after this one.
            # The minimum increment is 1.
            # So max_val + (m - pos - 1) <= max_scope => max_val <= max_scope - (m - pos - 1)
            max_val = max_scope - (m - pos - 1)
            
            if start_val > max_val:
                return

            for x in range(start_val, max_val + 1):
                # Calculate new diffs
                new_diffs = {x - a for a in current_row}
                # Check if new_diffs intersects with existing diffs
                if new_diffs & current_diffs:
                    continue
                # Also check if new_diffs has duplicates (shouldn't happen if x is unique relative to current_row, 
                # but x-a are distinct because a are distinct)
                
                # Recurse
                backtrack(pos + 1, current_row + [x], current_diffs | new_diffs)

        backtrack(0, [0], set())
        return valid_rows

    # --- Maximum Clique Solver using Bitsets ---
    def max_clique(adjacency, n, target_size):
        """
        Find a clique of size target_size in a graph with n nodes.
        adjacency[i] is a bitmask of neighbors of i.
        Returns list of node indices in the clique or None.
        """
        best_clique = []
        
        def bron_kerbosch(R, P, X):
            nonlocal best_clique
            if time.time() > deadline - 5:
                return True # Timeout signal
            if len(R) >= target_size:
                best_clique = R[:]
                return True
            
            if not P:
                return False
                
            # Pivot selection: choose vertex with max degree in P
            # In bitset representation, degree is popcount(adjacency[v] & P)
            pivot = -1
            max_deg = -1
            # Iterate through bits in P to find pivot
            p_bits = P
            while p_bits:
                v = (p_bits & -p_bits).bit_length() - 1
                deg = (adjacency[v] & P).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    pivot = v
                p_bits &= p_bits - 1
                
            # Candidates are P \ N(pivot)
            if pivot != -1:
                candidates = P & ~adjacency[pivot]
            else:
                candidates = P
                
            while candidates:
                v = (candidates & -candidates).bit_length() - 1
                candidates &= candidates - 1
                
                # Recurse
                if bron_kerbosch(R + [v], P & adjacency[v], X & adjacency[v]):
                    return True
                P &= ~(1 << v)
                X |= (1 << v)
                
            return False

        initial_P = (1 << n) - 1
        if bron_kerbosch([], initial_P, 0):
            return best_clique
        return None

    # --- Strategy 1: Clique-based Search ---
    # We will search for the minimum scope M such that a clique of size n_rows exists
    # in the compatibility graph of valid rows with max entry <= M.
    
    # Lower bound for M:
    # Total distinct differences needed is n_rows * diffs_per_row.
    # The differences must be distinct integers >= 1.
    # So max difference >= total_diffs_needed.
    # Since max difference in a row <= M, we need M >= total_diffs_needed.
    # Actually, the largest difference is max(row) - 0 = max(row).
    # So M >= total_diffs_needed is a loose lower bound.
    # A tighter bound: The sum of all differences in all rows is fixed? No.
    # Simple bound: M >= total_diffs_needed.
    
    lower_bound = total_diffs_needed
    # For n_rows=4, k=7, diffs_per_row=28, total=112.
    # Best known is 134.
    
    # Upper bound: Start with a greedy solution to get an initial upper bound.
    def greedy_initial(rng):
        used = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            cur = set()
            for _ in range(k):
                x = row[-1] + 1
                while True:
                    new_d = {x - a for a in row}
                    if not (new_d & used) and not (new_d & cur):
                        break
                    x += 1
                row.append(x)
                cur |= new_d
            rows.append(row)
            used |= cur
        return rows

    rng = random.Random(42)
    init_rows = greedy_initial(rng)
    upper_bound = scope_of(init_rows)
    
    best_rows = init_rows
    best_scope = upper_bound

    # Try to improve using clique search
    # We iterate M from lower_bound up to upper_bound
    # If we find a clique, we have a valid solution with scope <= M.
    # We want the smallest M.
    
    # Optimization: Instead of generating rows for every M, we can generate rows for the current upper_bound
    # and then filter/prune. But generating for smaller M is safer for correctness.
    
    # To save time, we can try a few specific M values.
    # Or, we can binary search if the property is monotonic (it is: if solution exists for M, it exists for M+1).
    
    # However, generating rows is expensive.
    # Let's try a linear scan from lower_bound upwards, but stop if we exceed upper_bound or time.
    # Actually, binary search is better.
    
    lo = lower_bound
    hi = upper_bound
    found_solution = None
    
    # We need a function that checks if a clique exists for a given M.
    def check_clique(M):
        if time.time() > deadline - 10:
            return None
        rows = generate_valid_rows(M)
        if not rows:
            return None
        
        n = len(rows)
        if n < n_rows:
            return None
            
        # Precompute diff sets
        diff_sets = [row_diffs(r) for r in rows]
        
        # Build adjacency matrix as bitsets
        # adjacency[i] has bit j set if rows[i] and rows[j] are compatible (no shared diffs)
        adjacency = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if not (diff_sets[i] & diff_sets[j]):
                    adjacency[i] |= (1 << j)
                    adjacency[j] |= (1 << i)
                    
        # Find max clique
        clique = max_clique(adjacency, n, n_rows)
        if clique:
            return [rows[i] for i in clique]
        return None

    # Binary search for the smallest M
    # To avoid expensive generation in every step, we might just try a few values.
    # Given the tight time limit, a full binary search with row generation might be too slow
    # if row generation is O(M^k).
    
    # Let's try a few candidates:
    # 1. Lower bound
    # 2. Best known (if applicable, e.g., 134 for 4,7)
    # 3. Current best scope
    # 4. Midpoints
    
    candidates_M = set()
    candidates_M.add(lo)
    candidates_M.add(hi)
    candidates_M.add(lo + (hi - lo) // 2)
    candidates_M.add(lo + (hi - lo) // 4)
    candidates_M.add(lo + 3 * (hi - lo) // 4)
    
    # Sort candidates
    for M in sorted(candidates_M):
        if time.time() > deadline - 20:
            break
        sol = check_clique(M)
        if sol:
            new_scope = scope_of(sol)
            if new_scope < best_scope:
                best_rows = sol
                best_scope = new_scope
                # If we found a solution, we can try to go lower
                hi = M - 1
                if lo >= hi:
                    break
                # Update candidates for next iteration? 
                # For simplicity, we break and rely on the fact that we found a good one.
                # Or we can continue to search for even lower M in the remaining time.
                # Let's add new candidates around the found M.
                candidates_M = {M - 1, M - 2, M - 5, M - 10}
                # We need to restart the loop or just check these.
                # Let's just check these specific ones.
                for M2 in sorted(candidates_M):
                    if M2 < lo:
                        continue
                    if time.time() > deadline - 10:
                        break
                    sol2 = check_clique(M2)
                    if sol2:
                        s2 = scope_of(sol2)
                        if s2 < best_scope:
                            best_rows = sol2
                            best_scope = s2
                break

    # --- Strategy 2: Fallback Heuristics (if clique search didn't improve or failed) ---
    # If best_scope is still the initial greedy one, or if we have time, run the old heuristics.
    
    def tighten(rows, max_iter=100):
        rows = [list(r) for r in rows]
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            improved = False
            for r in range(n_rows):
                row = rows[r]
                for pos in range(len(row) - 1, 0, -1):
                    target = row[pos - 1] + 1
                    if target >= row[pos]:
                        continue
                    cand = row[:pos] + [target] + row[pos + 1:]
                    ok = True
                    for i in range(1, len(cand)):
                        if cand[i] <= cand[i - 1]:
                            ok = False
                            break
                    if not ok:
                        continue
                    idiffs = set()
                    for i in range(len(cand)):
                        for j in range(i + 1, len(cand)):
                            d = cand[j] - cand[i]
                            if d in idiffs:
                                ok = False
                                break
                            idiffs.add(d)
                        if not ok:
                            break
                    if not ok:
                        continue
                    conflict = False
                    for idx in range(n_rows):
                        if idx == r:
                            continue
                        od = row_diffs(rows[idx])
                        if idiffs & od:
                            conflict = True
                            break
                    if not conflict:
                        row[pos] = target
                        improved = True
            if not improved:
                break
        return rows

    # Run tighten on the best found so far
    if time.time() < deadline - 5:
        tightened = tighten(best_rows)
        if is_valid(tightened):
            s = scope_of(tightened)
            if s < best_scope:
                best_rows = tightened
                best_scope = s

    # If we still have time and haven't found a solution close to the bound, try a quick anneal
    if time.time() < deadline - 10 and best_scope > lower_bound + 5:
        def anneal(rows, rng, max_iter=5000):
            rows = [list(r) for r in rows]
            used = set()
            for r in rows:
                used |= row_diffs(r)
            current_max = scope_of(rows)
            best_max = current_max
            best_rows = [list(r) for r in rows]
            T = 5.0
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    break
                r_idx = rng.randint(0, n_rows - 1)
                old_row = rows[r_idx]
                old_diffs = row_diffs(old_row)
                used -= old_diffs
                
                # Regenerate row
                new_row = [0]
                cur = set()
                ok_gen = True
                for _ in range(k):
                    x = new_row[-1] + 1
                    while True:
                        new_d = {x - a for a in new_row}
                        if not (new_d & used) and not (new_d & cur):
                            break
                        x += 1
                        if x > 500: # Cap
                            ok_gen = False
                            break
                    if not ok_gen:
                        break
                    new_row.append(x)
                    cur |= new_d
                if not ok_gen:
                    used |= old_diffs
                    continue
                new_diffs = row_diffs(new_row)
                other_max = max(rows[i][-1] for i in range(n_rows) if i != r_idx)
                new_global = max(other_max, new_row[-1])
                delta = new_global - current_max
                if delta < 0 or (delta == 0 and rng.random() < 0.5) or (rng.random() < math.exp(-delta / max(T, 0.1))):
                    rows[r_idx] = new_row
                    used |= new_diffs
                    current_max = new_global
                    if current_max < best_max:
                        best_max = current_max
                        best_rows = [list(r) for r in rows]
                else:
                    used |= old_diffs
                T *= 0.999
            return best_rows

        rows_ann = anneal(best_rows, rng, max_iter=2000)
        if is_valid(rows_ann):
            s_ann = scope_of(rows_ann)
            if s_ann < best_scope:
                best_rows = rows_ann
                best_scope = s_ann
                best_rows = tighten(best_rows)
                if is_valid(best_rows):
                    best_scope = scope_of(best_rows)

    return best_rows

if __name__ == "__main__":
    sol = construct(4, 7)
    used = set()
    valid = True
    for row in sol:
        if len(row) != 8 or row[0] != 0:
            valid = False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    valid = False
                used.add(d)
    print("n_rows=4 k=7 valid:", valid, "scope:", max(max(r) for r in sol))
