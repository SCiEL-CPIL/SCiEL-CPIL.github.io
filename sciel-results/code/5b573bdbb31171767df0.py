"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct a difference triangle set with n_rows rows, each of length k+1,
    starting at 0, strictly increasing, such that all positive differences
    within each row are distinct across all rows.
    
    Strategy: Binary search on the maximum scope M. 
    For each M, use a randomized greedy algorithm to try to construct the rows.
    If successful, we have a valid set. We minimize M.
    """
    
    def try_construct(m, seed):
        """
        Try to construct n_rows rows of k+1 elements in [0, m] 
        with distinct differences.
        Returns rows if successful, None otherwise.
        """
        random.seed(seed)
        used_diffs = set()
        rows = []
        
        # We will construct rows one by one.
        # To increase diversity, we can shuffle the order of rows or use different random seeds for each row attempt.
        
        for _ in range(n_rows):
            row = [0]
            # Greedy with backtracking? 
            # Simple greedy: pick smallest valid x.
            # Randomized greedy: pick from a set of candidates.
            
            # Let's use a simple greedy first, then if it fails, try random restarts.
            # Actually, for a fixed M, we can just try to fill it.
            
            success = False
            for attempt in range(50): # 50 random restarts for this row
                current_row = [0]
                valid = True
                for pos in range(1, k + 1):
                    # Find all valid next values
                    # Must be > current_row[-1] and <= m
                    # And diffs with all previous elements in current_row must be in used_diffs? No, must NOT be in used_diffs.
                    
                    candidates = []
                    start = current_row[-1] + 1
                    end = m + 1
                    
                    # Optimization: If the range is large, we can't iterate all.
                    # But m is around 100-200. k is 8.
                    # So we can iterate.
                    
                    for x in range(start, end):
                        # Check diffs with current_row
                        ok = True
                        for prev in current_row:
                            d = x - prev
                            if d in used_diffs:
                                ok = False
                                break
                        if ok:
                            candidates.append(x)
                    
                    if not candidates:
                        valid = False
                        break
                    
                    # Pick a candidate
                    # Heuristic: pick smallest to leave room? Or random?
                    # To help future rows, maybe picking smaller is better?
                    # Or picking larger?
                    # Let's try picking randomly from the first 10 or so, or just smallest.
                    # Smallest is standard greedy. Let's add randomness.
                    if len(candidates) > 1:
                        # Pick randomly from top 5 smallest or all?
                        # Let's pick randomly from all candidates to explore space
                        x = random.choice(candidates)
                    else:
                        x = candidates[0]
                    
                    # Before committing, we must check if this choice leads to a dead end later?
                    # We don't know yet.
                    current_row.append(x)
                    # Update used_diffs tentatively? No, we commit only if the whole row is built.
                    # But we need to check conflicts with *this* row's existing diffs too?
                    # The check `d in used_diffs` only checks against global used_diffs.
                    # We must also ensure diffs within current_row are distinct.
                    # Since we pick x > prev, and we checked d against used_diffs (which doesn't include current_row's internal diffs yet),
                    # we need to check against current_row's internal diffs.
                    
                    # Let's refine the candidate generation to include internal checks.
                    
                    # Restart the inner loop logic properly.
                    # This attempt loop is getting messy. Let's use a cleaner backtracking or recursive helper.
                    
                if valid:
                    # Check if the whole row is valid with itself
                    row_diffs = set()
                    for i in range(len(current_row)):
                        for j in range(i + 1, len(current_row)):
                            d = current_row[j] - current_row[i]
                            if d in row_diffs:
                                valid = False
                                break
                            row_diffs.add(d)
                    if not valid:
                        continue
                    
                    # Check if row_diffs intersects with global used_diffs
                    if row_diffs & used_diffs:
                        continue
                    
                    # Commit
                    rows.append(current_row)
                    used_diffs |= row_diffs
                    success = True
                    break
            
            if not success:
                return None
        
        return rows

    def get_scope(rows):
        return max(max(row) for row in rows)

    # Binary search for the minimum M
    # Lower bound: counting argument.
    # Number of diffs = n_rows * k * (k+1) / 2
    # Diffs are in [1, M]
    # n_rows * k * (k+1) / 2 <= M
    # 4 * 8 * 9 / 2 = 144
    # So M >= 144? No, diffs can repeat in value? No, "distinct over all rows".
    # So all 144 diffs must be distinct integers in [1, M].
    # So M >= 144.
    # Wait, the problem statement says "best known set has largest entry 134".
    # This implies M < 144 is possible?
    # Ah, the differences are $a_j - a_i$.
    # If the largest entry is $M$, the largest difference is $M$.
    # So the differences are a subset of $\{1, 2, \dots, M\}$.
    # There are $N_{diffs} = n\_rows \times \binom{k+1}{2}$ differences.
    # They must be distinct.
    # So $N_{diffs} \le M$.
    # $4 \times 36 = 144$.
    # So $M \ge 144$.
    # Why does the prompt say 134?
    # "The best known set has largest entry 134".
    # This contradicts the simple counting bound $M \ge 144$.
    # Let me re-read carefully.
    # "all differences between two entries of the same row are distinct over all rows"
    # Yes, global distinctness.
    # Is it possible that the prompt implies a different definition?
    # Or maybe I calculated $\binom{9}{2}$ wrong?
    # $9 \times 8 / 2 = 36$. Correct.
    # $4 \times 36 = 144$.
    # So $M$ must be at least 144.
    # The prompt says "simple counting bound is 112".
    # $112 < 144$. This suggests the counting bound calculation in the prompt is different or I am misunderstanding "differences".
    # Maybe "differences" means $|a_j - a_i|$? Yes, positive differences.
    # Maybe the rows don't have to be distinct from each other in differences?
    # "distinct over all rows" implies global distinctness.
    
    # Let's check the source "Shehadeh, Kingsford and Kschischang, 2025".
    # If the prompt is correct, then $M=134$ is possible.
    # This implies $N_{diffs} \le 134$.
    # $4 \times 36 = 144$.
    # There is a contradiction.
    # Unless... the differences within a row don't have to be distinct?
    # "all differences between two entries of the same row are distinct over all rows"
    # This phrasing is ambiguous.
    # Interpretation A: The set of all differences $\{a_{r,j} - a_{r,i} \mid \forall r, i<j\}$ has size equal to the number of pairs. (Global distinctness).
    # Interpretation B: For each row, the differences are distinct. AND the sets of differences for different rows are disjoint?
    # That's the same as A.
    
    # What if "distinct over all rows" means that for a specific difference value $d$, it can appear in at most one row?
    # That's still global distinctness.
    
    # Is it possible that $k$ is not 8?
    # "Choose 4 rows of 8 increasing integers starting at 0"
    # "Each row is a list of k + 1 integers"
    # If it says "8 increasing integers", maybe it means 8 non-zero integers?
    # "starting at 0" -> $x_0=0$.
    # "8 increasing integers" -> $x_1, \dots, x_8$?
    # Then total elements = 9.
    # If it meant 8 elements total (including 0), then $k=7$.
    # $\binom{8}{2} = 28$.
    # $4 \times 28 = 112$.
    # This matches the "simple counting bound is 112".
    # So, the task is for **8 elements per row** (including 0), i.e., $k=7$.
    # But the interface is `construct(n_rows, k)`.
    # The prompt says: "The evaluator passes this task's n_rows and k."
    # And "Each row is a list of k + 1 integers".
    # If the task is "Difference triangle set (4,7)", it likely means $n=4, k=7$.
    # Let's assume $k$ passed by evaluator is 7.
    # If $k=7$, elements are $0, x_1, \dots, x_7$. Total 8 elements.
    # Diffs per row: $\binom{8}{2} = 28$.
    # Total diffs: $4 \times 28 = 112$.
    # Min scope $M \ge 112$.
    # Best known 134.
    
    # So my code must handle the $k$ passed in.
    
    low = n_rows * (k * (k + 1) // 2)
    high = 200 # Arbitrary high bound
    
    best_rows = None
    
    # We can just iterate M from low upwards?
    # Or binary search.
    # Since the search space is small, we can try M from low to low + 50.
    
    for m in range(low, high):
        # Try multiple seeds for this m
        found = False
        for seed in range(100):
            rows = try_construct(m, seed)
            if rows is not None:
                best_rows = rows
                found = True
                break
        if found:
            return best_rows
            
    # Fallback: return the greedy construction if no valid set found in range
    # This should not happen if high is large enough
    
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
        
    return greedy_construct()

# EVOLVE-BLOCK-END
