"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _precompute_conflicts():
    """
    Precompute all 4-subsets of [1..16] and the conflict graph.
    Two sets conflict if there exists a third set that, together with them,
    forms a sunflower (all three pairwise intersections equal).
    """
    n = 16
    k = 4
    all_sets = list(combinations(range(1, n+1), k))
    total = len(all_sets)  # C(16,4) = 1820
    
    # Map each set to an index
    set_to_idx = {s: i for i, s in enumerate(all_sets)}
    
    # For each pair of sets, determine if they are in conflict.
    # A conflict exists if there is some set C such that:
    #   A ∩ B == A ∩ C == B ∩ C
    # This is the sunflower condition for 3 petals.
    
    # Efficient approach: for each pair (A, B), compute their intersection I = A ∩ B.
    # Then we need to check if there exists C ≠ A, B such that A ∩ C = I and B ∩ C = I.
    # This means C must contain I, and C must not contain any element from (A \ I) or (B \ I).
    # So C ⊇ I and C ∩ (A \ I) = ∅ and C ∩ (B \ I) = ∅.
    # Since |A| = |B| = 4 and I = A ∩ B, we have |A \ I| = 4 - |I| and |B \ I| = 4 - |I|.
    # The elements available for C (beyond I) are from [1..16] \ (A ∪ B).
    # C must be a 4-set containing I. So C = I ∪ D where D is a (4 - |I|)-subset of [1..16] \ (A ∪ B).
    # But we also need A ∩ C = I and B ∩ C = I.
    # A ∩ C = A ∩ (I ∪ D) = I ∪ (A ∩ D). Since D ⊆ [1..16] \ (A ∪ B), A ∩ D = ∅. So A ∩ C = I. Good.
    # Similarly B ∩ C = I. Good.
    # So any C of the form I ∪ D where D is a (4-|I|)-subset of [1..16] \ (A ∪ B) works.
    # We need at least one such C that is not A or B.
    # Note: if |I| = 4, then A = B, but we only consider distinct pairs.
    # If |I| < 4, then C ≠ A and C ≠ B automatically since C contains I and D, while A contains A\I which is non-empty.
    # Wait, C could equal A if D = A \ I, but D ⊆ [1..16] \ (A ∪ B), so D cannot contain elements of A. Thus C ≠ A.
    # Similarly C ≠ B.
    # So the condition is: there exists a (4 - |I|)-subset D of [1..16] \ (A ∪ B).
    # This is possible iff |[1..16] \ (A ∪ B)| >= 4 - |I|.
    # |A ∪ B| = |A| + |B| - |I| = 8 - |I|.
    # |[1..16] \ (A ∪ B)| = 16 - (8 - |I|) = 8 + |I|.
    # We need 8 + |I| >= 4 - |I|, i.e., 2|I| >= -4, which is always true.
    # So for ANY distinct pair A, B with |A ∩ B| < 4, there exists a C forming a sunflower!
    # Wait, that can't be right. Let me re-examine.
    
    # Actually, the sunflower condition is that all three pairwise intersections are equal.
    # For A, B, C: A∩B = A∩C = B∩C.
    # If A∩B = I, then we need A∩C = I and B∩C = I.
    # As shown, any C = I ∪ D with D ⊆ [1..16]\(A∪B), |D| = 4-|I|, works.
    # The number of such C is C(8+|I|, 4-|I|).
    # For this to be >= 1, we need 8+|I| >= 4-|I|, i.e., 2|I| >= -4, always true.
    # So indeed, for any distinct A, B, there exists a C such that A, B, C form a sunflower.
    # This means NO two distinct 4-subsets can coexist in the family!
    # But that contradicts the task which says max 250.
    
    # Let me re-read the task: "No three distinct sets may have all three pairwise intersections equal."
    # So the family is a set of 4-subsets where no three form a sunflower.
    # If every pair is in conflict, the max family size is 1. But the seed program returns more.
    
    # I must have made an error. Let's check with a small example.
    # A = {1,2,3,4}, B = {1,2,5,6}. I = {1,2}. |I| = 2.
    # C must contain {1,2} and be a 4-set, with C ∩ A = {1,2} and C ∩ B = {1,2}.
    # So C = {1,2,x,y} where x,y ∉ A ∪ B = {1,2,3,4,5,6}.
    # x,y from {7,8,...,16}. C(10,2) = 45 choices. All valid.
    # So A and B are in conflict.
    
    # What if A and B are disjoint? A = {1,2,3,4}, B = {5,6,7,8}. I = ∅.
    # C must be a 4-set with C ∩ A = ∅ and C ∩ B = ∅. So C ⊆ {9,...,16}.
    # C(8,4) = 70 choices. All valid.
    # So A and B are in conflict.
    
    # It seems like every pair is in conflict. So the max family size should be 1.
    # But the seed program returns a large family. Let me check the seed's condition.
    # Seed: all(not(a & b == a & s == b & s) for a,b in combinations(chosen,2))
    # This checks for each pair (a,b) in chosen, and for the new set s, whether a∩b == a∩s == b∩s.
    # So it's checking if adding s creates a sunflower with any existing pair (a,b).
    # This is correct.
    
    # So if every pair is in conflict, the seed should only return 1 set. But it returns more.
    # Let me run a quick mental test.
    # First set: A = {1,2,3,4}.
    # Second set: B = {1,2,5,6}. Check: no pairs in chosen (only A), so add B.
    # Third set: C = {1,2,7,8}. Check pairs (A,B): A∩B = {1,2}. A∩C = {1,2}. B∩C = {1,2}. All equal! So C is rejected.
    # What about C = {1,3,5,7}? A∩B = {1,2}. A∩C = {1,3}. Not equal. So C is accepted.
    # So the family is {A, B, C} = {{1,2,3,4}, {1,2,5,6}, {1,3,5,7}}.
    # Check all pairs:
    # (A,B): A∩B={1,2}, A∩C={1,3}, B∩C={5}. Not all equal. OK.
    # (A,C): A∩C={1,3}, A∩B={1,2}, C∩B={5}. Not all equal. OK.
    # (B,C): B∩C={5}, B∩A={1,2}, C∩A={1,3}. Not all equal. OK.
    # So the family is valid.
    
    # My earlier analysis was wrong. The conflict is not between a pair, but between a triple.
    # Two sets A, B are in "conflict" with a third set C if A∩B = A∩C = B∩C.
    # So the graph is not a simple conflict graph between pairs. It's a 3-uniform hypergraph of sunflowers.
    # We want the largest set of 4-subsets with no sunflower.
    
    # So the approach is: build a graph where vertices are 4-subsets, and an edge exists between A and B if there exists C such that A,B,C form a sunflower.
    # Then we want the maximum independent set.
    
    # But as shown, for ANY distinct A, B, there exists C such that A,B,C form a sunflower.
    # So the graph is complete, and the max independent set is 1.
    # But the example above shows a valid family of size 3.
    
    # I see the error: the sunflower condition is A∩B = A∩C = B∩C.
    # In the example, A={1,2,3,4}, B={1,2,5,6}, C={1,3,5,7}.
    # A∩B = {1,2}. A∩C = {1,3}. B∩C = {5}. These are NOT all equal.
    # So A, B, C do NOT form a sunflower.
    # My earlier claim that for any A, B there exists C such that A∩B = A∩C = B∩C was wrong.
    # The condition is that A∩C must equal A∩B, and B∩C must equal A∩B.
    # So C must intersect A exactly at A∩B, and intersect B exactly at A∩B.
    # This means C contains A∩B, and C is disjoint from A\(A∩B) and B\(A∩B).
    # So C ⊇ A∩B and C ∩ (A\(A∩B)) = ∅ and C ∩ (B\(A∩B)) = ∅.
    # So C = (A∩B) ∪ D where D ⊆ [1..16] \ (A ∪ B), |D| = 4 - |A∩B|.
    # The number of such C is C(16 - |A∪B|, 4 - |A∩B|) = C(16 - (8 - |A∩B|), 4 - |A∩B|) = C(8 + |A∩B|, 4 - |A∩B|).
    # For this to be >= 1, we need 8 + |A∩B| >= 4 - |A∩B|, i.e., 2|A∩B| >= -4, always true.
    # So there IS always such a C. But C must be distinct from A and B.
    # C = (A∩B) ∪ D. A = (A∩B) ∪ (A\(A∩B)). Since D ⊆ [1..16]\(A∪B), D ∩ (A\(A∩B)) = ∅, so C ≠ A.
    # Similarly C ≠ B.
    # So for any distinct A, B, there exists C such that A, B, C form a sunflower.
    # But in the example, A={1,2,3,4}, B={1,2,5,6}, A∩B={1,2}.
    # C must be {1,2} ∪ D where D is a 2-subset of [1..16]\{1,2,3,4,5,6} = {7,...,16}.
    # So C = {1,2,7,8} is a sunflower with A and B.
    # But C={1,3,5,7} is NOT a sunflower with A and B, because A∩C={1,3}≠{1,2}.
    # So the family {A, B, C={1,3,5,7}} is valid because no three form a sunflower.
    # The conflict is: A and B are in conflict with C={1,2,7,8}, but not with C={1,3,5,7}.
    # So the "conflict" is not between pairs of sets, but the existence of a sunflower depends on the triple.
    
    # To build the conflict graph: two sets A and B are adjacent if there exists C such that A,B,C form a sunflower.
    # As shown, for any distinct A, B, such a C exists. So the conflict graph is complete.
    # But that would mean no two sets can coexist, which is false.
    
    # I think I'm confusing myself. The task is: no three distinct sets in the family may form a sunflower.
    # So if I have a family F, for any three distinct A, B, C in F, A∩B ≠ A∩C or A∩B ≠ B∩C or A∩C ≠ B∩C.
    # The seed program checks: for each pair (a,b) in chosen, and new set s, check if a∩b == a∩s == b∩s.
    # This is correct.
    
    # So the approach is: greedy algorithm with randomization and restarts.
    # Precompute for each pair (A, B) whether they are in conflict (i.e., there exists C such that A,B,C form a sunflower).
    # As shown, every pair is in conflict. So the conflict graph is complete.
    # But that doesn't mean we can't have a large family. It means that for any pair A, B in the family, we must ensure that no C in the family forms a sunflower with A and B.
    # So when adding a new set C, we must check all pairs (A, B) in the current family and ensure that A∩B ≠ A∩C and A∩B ≠ B∩C and A∩C ≠ B∩C.
    # Equivalently, A∩B, A∩C, B∩C are not all equal.
    
    # So the algorithm is:
    # 1. Precompute all 4-subsets.
    # 2. Use a greedy algorithm with randomization: repeatedly pick a random set, add it if it doesn't form a sunflower with any existing pair.
    # 3. Use multiple restarts to find a larger family.
    
    # To speed up, we can precompute for each set C, the sets of pairs (A, B) that would form a sunflower with C.
    # But that's expensive. Instead, we can precompute for each pair (A, B), the intersection I = A∩B, and then when adding C, check if A∩C == I and B∩C == I.
    
    # Let's implement a greedy algorithm with bitmasks for speed.
    
    n = 16
    k = 4
    all_sets = list(combinations(range(n), k))  # 0-indexed
    total = len(all_sets)
    
    # Convert to bitmasks
    masks = [0] * total
    for i, s in enumerate(all_sets):
        m = 0
        for x in s:
            m |= (1 << x)
        masks[i] = m
    
    # Precompute intersections as bitmasks
    # inter[i][j] = masks[i] & masks[j]
    # This is a 1820x1820 matrix, which is ~3.3M entries. Each is an int. Should be fine.
    
    # For the greedy algorithm, we need to check if adding set C creates a sunflower with any pair (A, B) in the current family.
    # Condition: inter[A][B] == inter[A][C] and inter[A][B] == inter[B][C].
    # Equivalently, inter[A][B] == inter[A][C] == inter[B][C].
    
    # To speed up, for each set C, we can precompute the set of pairs (A, B) that would form a sunflower with C.
    # But that's expensive. Instead, we can iterate over pairs in the current family and check.
    
    # Given the time limit, let's use a simple greedy with randomization and multiple restarts.
    
    def greedy_random(seed_val):
        rng = random.Random(seed_val)
        order = list(range(total))
        rng.shuffle(order)
        
        chosen = []
        chosen_set = set()
        
        for idx in order:
            m = masks[idx]
            ok = True
            for i in range(len(chosen)):
                for j in range(i+1, len(chosen)):
                    a = chosen[i]
                    b = chosen[j]
                    ab = masks[a] & masks[b]
                    ac = masks[a] & m
                    bc = masks[b] & m
                    if ab == ac and ab == bc:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                chosen.append(idx)
                chosen_set.add(idx)
        
        return chosen
    
    # Run multiple restarts
    best = []
    deadline = time.time() + 80  # 80 seconds, leaving 10s margin
    
    num_restarts = 100
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        result = greedy_random(r)
        if len(result) > len(best):
            best = result
    
    # Convert back to sorted lists of 1-indexed elements
    result = []
    for idx in best:
        s = all_sets[idx]
        result.append(sorted(x+1 for x in s))
    
    return result

def construct():
    return _precompute_conflicts()
# EVOLVE-BLOCK-END
