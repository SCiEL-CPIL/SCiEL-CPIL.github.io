import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    N = 192
    MAX_K = 48
    
    # Precompute all forbidden hyperedges.
    # A hyperedge is a set {a, b, c, d} such that a divides b+c+d.
    # We want to select a set S such that no element a in S is part of a hyperedge 
    # where the other 3 elements are also in S.
    # Actually, the condition is: for any a in S, there do not exist distinct b,c,d in S\{a} 
    # such that a | (b+c+d).
    # This is equivalent to: The set S does not contain any 4-tuple {a,b,c,d} where a|b+c+d.
    # Note: b,c,d are distinct and distinct from a.
    
    # To speed up checks, we can precompute for each pair (b,c) the set of 'a's that are forbidden 
    # if d is chosen? No, that's too complex.
    # Instead, we will use a fast validity check during the greedy phase.
    
    def is_valid_addition(candidate, current_set):
        """
        Checks if 'candidate' can be added to 'current_set' without violating the condition.
        1. candidate must not divide sum of any 3 in current_set.
        2. No element in current_set must divide sum of candidate + any 2 others in current_set.
        """
        # Condition 1: candidate | sum(3 from current_set)
        if len(current_set) >= 3:
            for c1, c2, c3 in combinations(current_set, 3):
                if (c1 + c2 + c3) % candidate == 0:
                    return False
        
        # Condition 2: a in current_set | (candidate + b + c) for b,c in current_set\{a}
        # This is equivalent to: exists a, b, c in current_set such that a != b, a != c, b != c
        # and a | (candidate + b + c)
        # Note: b and c are distinct from a.
        
        # Optimization: 
        # For each a in current_set, we need to check if there exist b, c in current_set \ {a}
        # such that (b + c) % a == (-candidate) % a.
        
        # To make this faster, we can precompute pair sums modulo a? 
        # Since current_set size is <= 48, and we do this for each candidate, 
        # O(N^3) per check might be too slow if we do many random restarts.
        # Let's optimize Condition 2.
        
        for a in current_set:
            target = (-candidate) % a
            # We need b, c in current_set, b != a, c != a, b != c, such that (b+c) % a == target.
            # Let S_a = {x in current_set | x != a}
            # We can iterate pairs in S_a.
            # Size of S_a is at most 47. Pairs ~ 1000.
            # Total loops: 48 * 1000 = 48,000 ops per candidate check.
            # With 192 candidates, that's ~9M ops per full scan. 
            # This is acceptable in Python if we are careful, but let's try to be smarter.
            
            # Alternative: Use a set of residues?
            # For a fixed a, we want to know if there are two distinct elements b, c in S_a
            # such that (b+c) % a == target.
            # We can iterate b in S_a, and check if (target - b) % a is present in S_a \ {b}.
            # We need to be careful with b == c case, but b and c must be distinct elements.
            # If b == c, then 2b % a == target. But we need distinct elements.
            # So if we find a residue r such that there are at least two elements in S_a with residue r,
            # and 2r % a == target, it's valid.
            # Or if there are two distinct residues r1, r2 such that (r1+r2)%a == target and both exist in S_a.
            
            # Let's build a frequency map of residues for S_a.
            # Residues are in [0, a-1].
            # Since a <= 192, this is small.
            
            residues = {}
            for x in current_set:
                if x == a:
                    continue
                r = x % a
                residues[r] = residues.get(r, 0) + 1
            
            # Now check if target can be formed by sum of two distinct elements.
            # Case 1: b == c. Then 2*b % a == target.
            # We need at least 2 elements with residue r where 2*r % a == target.
            # Case 2: b != c. Then r1 + r2 = target (mod a).
            # We need r1 != r2, and both exist in residues.
            # Or r1 == r2 but we already covered that in Case 1? 
            # Wait, if r1 == r2, then b and c have same residue. 
            # If we have two distinct elements b, c with same residue r, then b!=c is satisfied.
            # So if residues[r] >= 2 and 2*r % a == target, it's valid.
            
            found = False
            for r, count in residues.items():
                if (2 * r) % a == target and count >= 2:
                    found = True
                    break
            
            if not found:
                # Check distinct residues
                # To avoid double counting and checking r1+r2 and r2+r1, we can just iterate all pairs of keys
                # But keys are small (<= a).
                keys = list(residues.keys())
                for i in range(len(keys)):
                    r1 = keys[i]
                    for j in range(i + 1, len(keys)):
                        r2 = keys[j]
                        if (r1 + r2) % a == target:
                            found = True
                            break
                    if found:
                        break
            
            if found:
                return False
        
        return True

    def get_conflict_degree(candidate, current_set):
        """
        Returns a measure of how many conflicts candidate would create if added.
        Actually, for the greedy step, we want to pick the candidate that adds the FEWEST 
        new conflicts to the *remaining* potential set? 
        A simpler heuristic: Pick the candidate with the smallest "degree" in the hypergraph 
        induced by the remaining candidates.
        However, calculating exact degree is expensive.
        
        Alternative heuristic: 
        1. Start with an empty set.
        2. Identify all valid candidates.
        3. Among valid candidates, pick the one that minimizes the number of other candidates it "blocks" 
           (i.e., candidates that would become invalid if this one is added).
        This is O(N^2 * CostCheck) per step.
        """
        pass

    def greedy_randomized():
        """
        Randomized greedy algorithm.
        """
        remaining = set(range(1, N + 1))
        selected = []
        
        # To make it deterministic-ish but varied, we can shuffle the initial order 
        # or use random tie-breaking.
        
        while remaining and len(selected) < MAX_K:
            valid_candidates = []
            for x in remaining:
                if is_valid_addition(x, selected):
                    valid_candidates.append(x)
            
            if not valid_candidates:
                break
            
            # Heuristic: Pick the candidate that leaves the most options for the future.
            # Approximation: Pick the one with the smallest "impact" on remaining set.
            # Impact = number of y in remaining\{x} that become invalid if x is added.
            
            # To save time, if valid_candidates is large, we might just pick randomly 
            # or pick the one with smallest value (or largest) to break ties.
            # Let's try a "minimum conflict" heuristic.
            # For each x in valid_candidates, count how many z in remaining\{x} 
            # would be invalid if x is in selected.
            # This is expensive: O(|valid| * |remaining| * CostCheck).
            # |valid| ~ 50, |remaining| ~ 100. CostCheck ~ 10k ops. 50*100*10k = 50M ops. Too slow.
            
            # Simpler heuristic: 
            # Large numbers are less likely to be divisors of sums of other numbers.
            # Small numbers are more likely to be divisors.
            # So, picking large numbers first is generally good.
            # But randomization helps escape local optima.
            
            # Let's use a weighted random choice favoring larger numbers?
            # Or just pick the largest valid candidate? (Current strategy)
            # To differ from current, let's pick randomly from the top 10% largest valid candidates.
            
            valid_candidates.sort(reverse=True)
            # Pick from top 10 or top 5%
            k = max(1, len(valid_candidates) // 10)
            choice = random.choice(valid_candidates[:k])
            
            selected.append(choice)
            remaining.remove(choice)
            
            # Check time
            if time.time() > deadline - 5:
                break
        
        return selected

    def local_search(initial_set):
        """
        Try to improve the set by swapping elements.
        1. Find an element x in selected that is "bad" (causes many conflicts).
        2. Find an element y not in selected that can replace x.
        """
        selected = set(initial_set)
        improved = True
        current_size = len(selected)
        
        # We try to increase size or maintain size with better "health".
        # Since we want max cardinality, we first try to add elements.
        # If we can't add, we try swaps to enable additions.
        
        # Phase 1: Try to add any valid element.
        # (This is basically what greedy did, but maybe greedy missed one due to order)
        # We'll rely on the greedy phase for additions.
        # Local search focuses on SWAPS to break deadlocks.
        
        # If we are stuck at size S, we try to swap x in S with y not in S,
        # such that the new set is valid, and then maybe we can add more?
        # Actually, a swap maintains size. 
        # To increase size, we need to find a y such that S + {y} is valid. 
        # If no such y exists, we must change S.
        # Swap x for y might open up space for z.
        
        # Algorithm:
        # While time allows:
        #   1. Check if any y in [1..N]\S can be added to S. If yes, add it and restart.
        #   2. If no, try a random swap: pick x in S, y not in S.
        #      If S - {x} + {y} is valid, accept it.
        #      Then go back to step 1.
        
        while time.time() < deadline - 5:
            # Step 1: Try to add
            added = False
            for y in range(1, N + 1):
                if y not in selected:
                    if is_valid_addition(y, list(selected)):
                        selected.add(y)
                        added = True
                        break # Restart loop to check for more additions
            if added:
                continue
            
            # Step 2: No addition possible. Try a swap.
            # Pick a random x from selected.
            x = random.choice(list(selected))
            # Pick a random y from not selected.
            # To be smarter, pick y that is "compatible" with most of S\{x}.
            # For now, random y.
            candidates_y = [y for y in range(1, N + 1) if y not in selected]
            if not candidates_y:
                break
            y = random.choice(candidates_y)
            
            # Check if S - {x} + {y} is valid.
            new_set = list(selected)
            new_set.remove(x)
            # Check validity of y in new_set
            if is_valid_addition(y, new_set):
                # Also need to check if removing x and adding y creates new conflicts 
                # among existing elements? No, existing elements were valid among themselves.
                # The only new conflicts involve y.
                # And conflicts involving x are removed.
                # So if y is valid in S\{x}, the new set is valid.
                selected.remove(x)
                selected.add(y)
                # After swap, loop back to try additions.
                continue
            
            # If too many failed swaps, break to avoid wasting time?
            # Or just continue.
        
        return list(selected)

    best_set = []
    
    # Run multiple randomized greedy restarts
    # Each restart takes some time. We need to balance number of restarts vs local search depth.
    
    num_restarts = 0
    while time.time() < deadline - 10:
        # Generate a random greedy solution
        greedy_sol = greedy_randomized()
        
        # Apply local search
        # Limit local search time per restart?
        # If greedy_sol is already 48, we can return immediately.
        if len(greedy_sol) >= MAX_K:
            return greedy_sol
        
        ls_sol = local_search(greedy_sol)
        
        if len(ls_sol) > len(best_set):
            best_set = ls_sol
            if len(best_set) >= MAX_K:
                return best_set
        
        num_restarts += 1
        # If we've done many restarts, maybe reduce local search time?
        # For now, fixed loop.
        
    return best_set

# Ensure we always return a valid set, even if empty or partial
# The function construct returns a list of integers.

if __name__ == "__main__":
    res = construct()
    print(f"Found set of size {len(res)}")
