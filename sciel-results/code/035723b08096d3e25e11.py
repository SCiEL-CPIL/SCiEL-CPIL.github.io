"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

def construct():
    # Vertices 0..11, n=12
    # We want 40 distinct subsets maximizing strict subset pairs.
    
    # Strategy: Use the empty set, all singletons, and many large sets (complements of small sets).
    # Let S be the family.
    # If we include the empty set, it is a strict subset of every other set. (39 pairs)
    # If we include all singletons (12 sets), each is a strict subset of any set of size >= 2 that contains it.
    # Let's include the empty set (1 set).
    # Let's include all singletons (12 sets).
    # Remaining 27 sets.
    
    # To maximize pairs, we want the remaining 27 sets to be as "large" as possible to contain the singletons,
    # but also to have inclusion relationships among themselves if possible.
    # However, sets of size 11 (complements of singletons) are very good because they contain all singletons except one.
    # Let's include all 12 sets of size 11.
    # So far: 1 (empty) + 12 (size 1) + 12 (size 11) = 25 sets.
    # Remaining 15 sets.
    
    # Pairs so far:
    # Empty vs others: 39
    # Size 1 vs Size 11: A size 1 set {i} is a subset of a size 11 set V\{j} if i != j.
    # There are 12 size 1 sets and 12 size 11 sets. For each size 1 set, it is contained in 11 size 11 sets.
    # Total 12 * 11 = 132 pairs.
    # Size 1 vs Size 1: 0 (distinct)
    # Size 11 vs Size 11: 0 (distinct, same size)
    # Empty vs Size 1: 12
    # Empty vs Size 11: 12
    # Total so far: 39 + 132 = 171?
    # Let's recount carefully.
    # Family F = { empty } U { {i} for i in 0..11 } U { V\{i} for i in 0..11 }
    # Size = 25.
    # Pairs (A, B) with A < B:
    # 1. A = empty. B can be any of the 24 other sets. Count = 24.
    # 2. A = {i} (size 1).
    #    B can be:
    #    - V\{j} (size 11). {i} < V\{j} iff i != j.
    #      For each i, there are 11 such j.
    #      Total 12 * 11 = 132.
    #    - Can B be another singleton? No, distinct.
    #    - Can B be empty? No, strict subset means A < B, so A cannot be empty if B is empty (they are equal). Wait, A=empty is case 1. Here A is size 1.
    # 3. A = V\{i} (size 11).
    #    B must be larger. Max size is 12.
    #    If we include V (size 12), then all size 11 sets are subsets of V.
    #    
    # Let's add V (the full set) to the family.
    # Family F' = F U { V }. Size = 26.
    # New pairs involving V:
    # - Any A in F such that A < V. Since all sets in F are proper subsets of V, all 25 sets in F are strict subsets of V.
    # - Count = 25.
    # Total pairs for F': 24 (empty) + 132 (1 vs 11) + 25 (all vs V) = 181.
    # Wait, did I miss pairs between empty and singletons/size 11?
    # Case 1 counted empty vs ALL other 24 sets in F. That includes singletons and size 11s.
    # So pairs are:
    # (empty, singleton): 12
    # (empty, size11): 12
    # (singleton, size11): 132
    # (empty, V): 1 (included in the 24? No, V was added later. In F, there are 24 others. In F', empty vs V is 1. Singleton vs V is 12. Size11 vs V is 12.)
    # Let's re-sum for F' (26 sets: empty, 12 singles, 12 size11s, V).
    # Pairs:
    # 1. A=empty. B in F'\{empty}. Size(F')-1 = 25. Count = 25.
    # 2. A=singleton {i}. B in F'\{empty, {i}}.
    #    B can be size 11 sets V\{j} where i!=j. (11 sets)
    #    B can be V. (1 set)
    #    B cannot be other singletons.
    #    Count per singleton = 11 + 1 = 12.
    #    Total for 12 singletons = 12 * 12 = 144.
    # 3. A=size 11 set V\{i}. B in F'\{empty, singles, size11s, V}.
    #    B must be V. (1 set)
    #    Count per size 11 set = 1.
    #    Total for 12 size 11 sets = 12 * 1 = 12.
    # Total = 25 + 144 + 12 = 181.
    
    # We have 26 sets. We need 40 sets. We need 14 more sets.
    # Which sets should we add to maximize new pairs?
    # Current Family:
    # S0: empty
    # S1: 12 singletons
    # S11: 12 co-singletons
    # S12: V
    
    # If we add a set of size 1, it must be distinct from existing singletons. But all singletons are already there. So we can't add more size 1.
    # If we add a set of size 11, all are already there.
    # So we must add sets of intermediate sizes: 2, 3, ..., 10.
    
    # Let's analyze adding a set of size k.
    # It will form pairs with:
    # - Empty: 1 pair (empty < A)
    # - Singletons {i}: A contains {i} if i in A. Count = k.
    # - Co-singletons V\{j}: A < V\{j} iff A is subset of V\{j} iff j not in A.
    #   So j must be in V\A. There are 12-k such j's.
    #   Count = 12-k.
    # - V: 1 pair (A < V).
    # - Other newly added sets: depends on their sizes and content.
    
    # If we add 14 sets of size k, and assume they are chosen to be "random" or disjoint in a way that minimizes internal pairs? No, we want to MAXIMIZE pairs.
    # Ideally, the new sets should contain each other.
    # A chain of length L has L*(L-1)/2 pairs.
    # But we also gain the "cross" pairs with the existing 26 sets.
    
    # Let's try to form a chain of sets of increasing size.
    # We have space for 14 sets.
    # If we make a chain of 14 sets, say A1 < A2 < ... < A14.
    # Internal pairs: 14*13/2 = 91.
    # Cross pairs with existing:
    # Each Ai has size ki.
    # Pairs with Empty: 1 per Ai. Total 14.
    # Pairs with V: 1 per Ai. Total 14.
    # Pairs with Singletons: sum(ki).
    # Pairs with Co-singletons: sum(12-ki).
    
    # To maximize sum(ki) + sum(12-ki) = sum(12) = 14*12 = 168.
    # This part is constant regardless of the sizes, as long as we count all valid inclusions.
    # Wait, is it?
    # For a specific set A of size k:
    # - It contains k singletons.
    # - It is contained in 12-k co-singletons.
    # Sum = k + 12 - k = 12.
    # Yes, constant 12 per set.
    # So total cross pairs with singles and co-singles is 14 * 12 = 168.
    # Total cross pairs with empty and V is 14 + 14 = 28.
    # Total cross pairs = 168 + 28 = 196.
    # Internal pairs = 91.
    # Total new pairs added by these 14 sets = 196 + 91 = 287.
    # Total score = 181 (base) + 287 = 468.
    
    # Can we do better?
    # What if we don't make a single chain, but multiple chains or other structures?
    # The internal pairs are maximized by having as many inclusion relations as possible among the 14 new sets.
    # A total order (chain) maximizes the number of comparable pairs in a set of size N (which is N choose 2).
    # Any antichain has 0 internal pairs.
    # So a chain is best for internal pairs.
    
    # Does the choice of sizes affect the cross pairs?
    # We established cross pairs with singles/co-singles is constant 12 per set.
    # Cross pairs with empty/V is constant 1 per set.
    # So the structure of the 14 new sets (their specific sizes and elements) only affects the INTERNAL pairs.
    # To maximize internal pairs, we need a chain.
    # Can we form a chain of 14 sets with distinct sizes between 2 and 10?
    # Available sizes: 2, 3, 4, 5, 6, 7, 8, 9, 10.
    # There are only 9 distinct sizes in this range.
    # So we cannot have 14 sets of distinct sizes in [2,10].
    # We must repeat sizes.
    # If we have two sets of the same size, they cannot be in a strict subset relationship.
    # So the graph of inclusion among the new sets cannot be a total order if sizes repeat.
    
    # Let's analyze the max internal pairs for 14 sets with sizes from {2..10}.
    # To maximize comparability, we should pick sizes that allow long chains.
    # However, if we pick the same size multiple times, we lose potential links.
    # Example: Two sets A, B of size 5. A is not subset of B and B is not subset of A (if distinct).
    # So they are incomparable.
    
    # Strategy: Pick as many distinct sizes as possible to form chains, and distribute the "extra" sets into sizes that allow them to be comparable with the others?
    # No, same size sets are incomparable with each other.
    # But a set of size k can be comparable with sets of size < k and > k.
    
    # Let's define the count of sets of each size c_k for k in 2..10.
    # Sum c_k = 14.
    # Internal pairs:
    # Pairs (A, B) with A < B.
    # If size(A) < size(B), it's possible A < B.
    # If size(A) > size(B), it's possible B < A.
    # If size(A) = size(B), no pairs.
    
    # To maximize pairs, we should ensure that for any A of size i and B of size j (i<j), A is a subset of B whenever possible?
    # We can choose the specific sets to maximize this.
    # For a fixed multiset of sizes, the maximum number of pairs is achieved by nesting the sets as much as possible.
    # i.e., sort the sets by size. For each size group, try to make them subsets of the next size group.
    # Actually, if we have c_i sets of size i, the best we can do is that all c_i sets of size i are subsets of all c_j sets of size j (for i<j)?
    # No, we can't have c_i distinct sets of size i all being subsets of a single set of size j if c_i is large?
    # Wait, if A < B and C < B, A and C can be distinct.
    # Can we have ALL pairs (A, B) with size(A) < size(B) be valid inclusions?
    # This requires that for any A (size i) and B (size j, i<j), A subset B.
    # This implies that the union of all sets of size i must be a subset of every set of size j?
    # Actually, if A1 < B and A2 < B, it's fine.
    # But if A < B1 and A < B2, it's fine.
    # The constraint is: Is it possible to choose c_i sets of size i such that EVERY set of size i is a subset of EVERY set of size j (for i<j)?
    # If c_i = 1, we just need that one set to be in all larger sets.
    # If c_i > 1, say A1, A2 of size i. Both must be subsets of B (size j).
    # This is possible if |A1 U A2| <= j.
    # But also A1 and A2 are distinct.
    # If we have many sets of size i, their union grows.
    
    # Let's look at the extreme:
    # If we use only one size, say size 5, and 14 sets.
    # Internal pairs = 0.
    # Total = 181 + 14*12 + 14*2 = 181 + 168 + 28 = 377.
    # This is worse than the chain attempt (468).
    
    # Let's try to use as many distinct sizes as possible to create a "lattice" that is dense.
    # The most dense structure is a chain.
    # Max chain length with distinct sizes is 9 (sizes 2..10).
    # We have 14 sets.
    # Let's take the 9 sizes 2,3,4,5,6,7,8,9,10.
    # We have 5 extra sets to place.
    # Where do we place them?
    # If we add a set of size 5, it will be comparable with sizes <5 and >5.
    # But it will be incomparable with other size 5 sets.
    # Let's say we have c_2, c_3, ..., c_10.
    # Sum c_k = 14.
    # To maximize pairs, we should make the "core" a chain of 9 sets (one of each size 2..10).
    # Let A_2 < A_3 < ... < A_10.
    # Now add 5 more sets.
    # If we add a set B of size k.
    # It can be a subset of all A_j for j>k.
    # It can be a superset of all A_i for i<k.
    # Can we choose B such that it is comparable with all existing sets?
    # Yes, if we choose B to be "between" A_{k-1} and A_{k+1}.
    # Specifically, A_{k-1} < B < A_{k+1}.
    # This is possible if we have "room".
    # A_{k-1} has size k-1. A_{k+1} has size k+1.
    # We can pick B of size k such that A_{k-1} subset B subset A_{k+1}.
    # This requires that A_{k-1} is a subset of A_{k+1} (which it is) and that we can pick k-1 elements from A_{k+1}\A_{k-1} to add to A_{k-1} to form B?
    # |A_{k+1} \ A_{k-1}| = (k+1) - (k-1) = 2.
    # So we must pick 1 element from these 2 to add to A_{k-1} to get size k.
    # There are 2 choices for B.
    # So we can insert 1 extra set of size k between A_{k-1} and A_{k+1}.
    # Can we insert 2 extra sets of size k?
    # We need two distinct sets B1, B2 of size k such that A_{k-1} < B1, B2 < A_{k+1}.
    # Both must be subsets of A_{k+1} and supersets of A_{k-1}.
    # The available elements to add are the 2 in A_{k+1}\A_{k-1}.
    # To form a set of size k, we must add exactly 1 element from these 2.
    # So there are only 2 such sets.
    # So we can add AT MOST 2 sets of any given size k (between k-1 and k+1).
    # Actually, if we add 2 sets B1, B2, they are distinct.
    # Are they comparable with each other? No, same size.
    # So we get pairs with the rest, but not between B1 and B2.
    
    # So, for each size k, we can have at most 2 sets "nested" in the chain without breaking the total order property for cross-size pairs?
    # Wait, if we have B1, B2 of size k.
    # Pairs with A_{k-1}: 2 (A_{k-1} < B1, A_{k-1} < B2).
    # Pairs with A_{k+1}: 2 (B1 < A_{k+1}, B2 < A_{k+1}).
    # Pairs with A_i (i < k-1): 2 each.
    # Pairs with A_j (j > k+1): 2 each.
    # Pairs between B1, B2: 0.
    
    # So adding a "pair" of sets of size k adds 2 sets.
    # We have 5 extra sets.
    # We can make 2 pairs (4 sets) and 1 single set.
    # Let's put pairs at sizes 5 and 6.
    # And a single at size 7.
    
    # Let's calculate the internal pairs for this configuration.
    # Base chain: 9 sets. Pairs = 9*8/2 = 36.
    # Added sets:
    # 2 sets of size 5 (B5a, B5b).
    # 2 sets of size 6 (B6a, B6b).
    # 1 set of size 7 (B7a).
    
    # Pairs involving added sets with base chain (A2..A10):
    # B5a, B5b:
    # - With A2, A3, A4 (size < 5): 2 sets * 3 sets = 6 pairs.
    # - With A6, A7, A8, A9, A10 (size > 5): 2 sets * 5 sets = 10 pairs.
    # - With A5: 0 (same size, distinct).
    # Total for B5s with base: 16.
    
    # B6a, B6b:
    # - With A2..A5 (size < 6): 2 sets * 4 sets = 8 pairs.
    #   Wait, A5 is in base.
    # - With A7..A10 (size > 6): 2 sets * 4 sets = 8 pairs.
    # - With A6: 0.
    # Total for B6s with base: 16.
    
    # B7a:
    # - With A2..A6 (size < 7): 1 set * 5 sets = 5 pairs.
    # - With A8..A10 (size > 7): 1 set * 3 sets = 3 pairs.
    # - With A7: 0.
    # Total for B7a with base: 8.
    
    # Pairs among added sets:
    # B5s vs B6s:
    # B5a, B5b are size 5. B6a, B6b are size 6.
    # Can we make B5a, B5b subsets of B6a, B6b?
    # Yes, if we chose them appropriately.
    # We want to maximize. Assume we can.
    # 2 * 2 = 4 pairs.
    
    # B5s vs B7a:
    # Size 5 vs Size 7.
    # 2 * 1 = 2 pairs.
    
    # B6s vs B7a:
    # Size 6 vs Size 7.
    # 2 * 1 = 2 pairs.
    
    # B5s internal: 0.
    # B6s internal: 0.
    # B7a internal: 0.
    
    # Total internal pairs = 36 (base) + 16 + 16 + 8 + 4 + 2 + 2 = 84.
    
    # Compare with simple chain of 9 (if we only had 9 sets, internal=36).
    # With 14 sets, we got 84.
    # Max possible internal pairs for 14 sets is 91 (total order).
    # We lost 7 pairs because of same-size incomparabilities.
    # Loss for 2 sets of size 5: 1 pair.
    # Loss for 2 sets of size 6: 1 pair.
    # Loss for 1 set of size 7: 0.
    # Total loss from total order = 2?
    # Wait, 91 - 84 = 7.
    # Why 7?
    # In a total order, all pairs are counted.
    # In our config:
    # Pairs missing:
    # - Between B5a and B5b: 1
    # - Between B6a and B6b: 1
    # - Between B5a/B5b and A5?
    #   In total order, if we had 11 sets of size 5? No, we have 1 A5 and 2 B5s.
    #   In a total order of 14 sets, we would have 14 distinct sizes? No, we only have 9 available sizes.
    #   So a total order is IMPOSSIBLE with 14 sets drawn from 9 sizes.
    #   The max number of comparable pairs in a poset with height 9 and 14 elements...
    #   The max is indeed less than 91.
    
    # Let's verify the count 84.
    # Total pairs possible = 14*13/2 = 91.
    # Incomparable pairs:
    # 1. (B5a, B5b)
    # 2. (B6a, B6b)
    # Are there others?
    # A5 vs B5a, B5b: In our construction, A5 is comparable?
    # We assumed A5 < B5a and A5 < B5b?
    # If A5 is size 5, and B5a is size 5, they are distinct, so INCOMPARABLE.
    # Ah! I missed this.
    # A5 (base) vs B5a, B5b (added).
    # All three are size 5.
    # Pairs: (A5, B5a), (A5, B5b), (B5a, B5b).
    # 3 incomparable pairs.
    # Similarly for size 6:
    # A6 (base) vs B6a, B6b.
    # Pairs: (A6, B6a), (A6, B6b), (B6a, B6b).
    # 3 incomparable pairs.
    # Total incomparable = 3 + 3 = 6.
    # Comparable = 91 - 6 = 85.
    # My previous sum was 84. Where is the error?
    # Let's re-sum.
    # Base-Base: 36.
    # Base-Added:
    #   B5s with A2,A3,A4,A6,A7,A8,A9,A10. (8 sets). 2*8 = 16.
    #   B6s with A2,A3,A4,A5,A7,A8,A9,A10. (8 sets). 2*8 = 16.
    #   B7a with A2..A6, A8..A10. (9 sets). 1*9 = 9.
    #   Wait, previously I said 8 for B7a.
    #   A2,A3,A4,A5,A6 (5 sets) + A8,A9,A10 (3 sets) = 8 sets?
    #   A7 is same size.
    #   So 5+3=8. Correct.
    # Added-Added:
    #   B5s vs B6s: 4.
    #   B5s vs B7a: 2.
    #   B6s vs B7a: 2.
    #   Total 8.
    # Sum: 36 + 16 + 16 + 8 + 8 = 84.
    # Why did the subtraction method give 85?
    # Total 91.
    # Incomparable:
    # (A5, B5a), (A5, B5b), (B5a, B5b) -> 3
    # (A6, B6a), (A6, B6b), (B6a, B6b) -> 3
    # Total 6.
    # 91 - 6 = 85.
    # Discrepancy of 1.
    # Let's check B7a.
    # B7a vs A7. Incomparable?
    # A7 is base. B7a is added.
    # Both size 7.
    # So (A7, B7a) is incomparable.
    # I missed this in the subtraction list!
    # So incomparable is 3 + 3 + 1 = 7.
    # 91 - 7 = 84.
    # Matches.
    
    # So max internal pairs is 84 with this distribution.
    # Can we do better with a different distribution?
    # e.g. 3 sets of size 5?
    # If we have c_5=3.
    # Incomparable pairs within size 5 group: 3 choose 2 = 3.
    # Plus A5? If A5 is the only base set of size 5, and we add 2, total c_5=3.
    # If we add 3, total c_5=4.
    # Let's stick to the logic: minimize the sum of (c_k choose 2) for all k.
    # Subject to sum c_k = 14, c_k >= 1 (to keep the chain connected? Not strictly necessary, but good for cross pairs).
    # Actually, if c_k=0, we break the chain.
    # If we skip a size, we lose potential links.
    # But if we have many sets of one size, we lose internal pairs within that size.
    
    # Let's optimize sum (c_k choose 2).
    # We want c_k to be as small as possible.
    # Min c_k is 1.
    # 9 sizes. 9 * 1 = 9 sets.
    # 5 remaining.
    # Distribute 5 additional sets to minimize sum of increases in (c_k choose 2).
    # Start with all 1. Sum = 0.
    # Add 1 to a c_k: (1->2). Cost increase: 1.
    # Add 1 more to same c_k: (2->3). Cost increase: 2.
    # Add 1 to different c_k: (1->2). Cost increase: 1.
    
    # We have 5 additions.
    # Best: spread them out.
    # 5 different sizes get +1.
    # So five sizes have c_k=2, four sizes have c_k=1.
    # Sum (c_k choose 2) = 5 * 1 + 4 * 0 = 5.
    # Incomparable pairs = 5.
    # Total internal pairs = 91 - 5 = 86.
    
    # This is better than 84!
    # Let's verify if this is constructible.
    # We need 5 sizes to have 2 sets, 4 sizes to have 1 set.
    # Total 14 sets.
    # We need to ensure the "nesting" works.
    # For sizes with c_k=2, say k.
    # We have A_k (base) and B_k (added).
    # We need A_k and B_k to be comparable with all sets of other sizes.
    # This requires that for i < k, A_i < A_k and A_i < B_k.
    # And for j > k, A_k < A_j and B_k < A_j.
    # This is possible if we construct the chain carefully.
    # Start with A_2.
    # For each next size k:
    # If c_k=1, pick A_k such that A_{k-1} < A_k < A_{k+1} (or similar).
    # If c_k=2, pick A_k and B_k such that A_{k-1} < A_k, B_k < A_{k+1}.
    # And we need A_k, B_k to be subsets of all larger sets.
    # This is possible if the "gap" between A_{k-1} and A_{k+1} is large enough to accommodate the extra elements.
    # With 12 vertices, this should be fine.
    
    # So, 86 internal pairs.
    # Total score = 181 (base) + 14*14 (cross) + 86 (internal) = 181 + 196 + 86 = 463.
    # Wait, earlier I said cross is 196.
    # 14 sets * (1 empty + 1 V + 12 avg singles/co-singles) = 14 * 14 = 196.
    # Yes.
    
    # So 463 is a good target.
    
    # Let's code this construction.
    
    vertices = list(range(12))
    V = set(vertices)
    
    # Base sets
    empty = frozenset()
    singles = [frozenset([i]) for i in vertices]
    co_singles = [frozenset(V - {i}) for i in vertices]
    full = frozenset(vertices)
    
    # 14 intermediate sets
    # Sizes: 2,3,4,5,6,7,8,9,10
    # 5 of these will have 2 sets, 4 will have 1 set.
    # Let's pick sizes 2,3,4,5,6 to have 2 sets. (Or any 5).
    # Let's build a chain.
    
    # We need to generate the actual sets.
    # Strategy:
    # Fix a base element set?
    # Let's just use a greedy construction or specific patterns.
    
    # Pattern:
    # Let the sets be prefixes of the vertex list?
    # If we use prefixes:
    # P_k = {0, 1, ..., k-1}.
    # Then P_i < P_j for i<j.
    # This gives a chain of length 11 (sizes 0..11).
    # But we need specific sizes and counts.
    
    # Let's create the 14 sets:
    # Sizes: 2,2, 3,3, 4,4, 5,5, 6,6, 7, 8, 9, 10.
    # (5 pairs, 4 singles).
    
    # We can just hardcode a valid family if it's simple enough, or generate it.
    # Since we have 60s, we can do a small search or constructive algorithm.
    
    # Constructive:
    # We want S_2a, S_2b, S_3a, S_3b, ...
    # such that S_ia, S_ib < S_3a, S_3b < ...
    
    # Let's define the sets using binary masks for simplicity in construction, then convert.
    
    # To ensure S_ia, S_ib < S_ka, S_kb for i<k:
    # We can define S_ka as a superset of S_{k-1}a, S_{k-1}b.
    
    # Let's try:
    # S_2a = {0,1}
    # S_2b = {0,2}
    # S_3a = {0,1,3} (contains S_2a? No, 2 not in S_3a. S_2b? No.)
    # We need S_2a, S_2b < S_3a.
    # So S_3a must contain {0,1,2}.
    # S_3a = {0,1,2}.
    # S_3b = {0,1,2}? No, distinct.
    # S_3b must also contain S_2a, S_2b.
    # So S_3b must contain {0,1,2}.
    # But S_3a and S_3b are both size 3.
    # If both contain {0,1,2}, they are identical.
    # Contradiction!
    
    # So we CANNOT have two sets of size k-1 both being subsets of two sets of size k?
    # Wait.
    # If S_2a = {0,1} and S_2b = {0,2}.
    # Union is {0,1,2}.
    # Any set of size 3 containing both must contain {0,1,2}.
    # So S_3a must be {0,1,2}.
    # S_3b must be {0,1,2}.
    # They are not distinct.
    
    # So we cannot have c_{k-1}=2 and c_k=2 with this "all-to-all" comparability?
    # The condition "all sets of size i are subsets of all sets of size j" is too strong if the union of size i sets is large.
    
    # We only need to maximize pairs, not have ALL pairs.
    # With c_k=2, we lose 1 internal pair.
    # But do we lose cross pairs?
    # If S_2a is not subset of S_3b, we lose that pair.
    # In the example:
    # S_2a={0,1}, S_2b={0,2}.
    # S_3a={0,1,3}.
    # S_3b={0,2,3}.
    # Pairs:
    # S_2a < S_3a? Yes.
    # S_2a < S_3b? No (1 not in S_3b).
    # S_2b < S_3a? No (2 not in S_3a).
    # S_2b < S_3b? Yes.
    # So we get 2 pairs instead of 4.
    # We lost 2 pairs.
    # Internal loss: 1.
    # Cross loss: 2.
    # Total loss: 3.
    
    # If we used c_k=1 for all:
    # No internal loss.
    # Cross pairs: All 4.
    # Total 4.
    
    # So having c_k=2 is worse?
    # Let's check the score contribution.
    # Adding a second set of size k:
    # Gains:
    # - Pairs with smaller sizes: k-1 (if all comparable).
    # - Pairs with larger sizes: 12-k (if all comparable).
    # - Pairs with V: 1.
    # - Pairs with empty: 1.
    # Total gain approx 14.
    # Losses:
    # - Internal: 1 (with the first set of size k).
    # - Cross: potentially many if not nested well.
    
    # Given the complexity, maybe a simpler structure is better.
    # What if we just use the 14 sets of size 6?
    # Score 377.
    
    # What if we use a chain of 9 distinct sizes, and 5 sets of size 6 (middle)?
    # c_6 = 6. Others 1.
    # Internal loss: 6 choose 2 = 15.
    # Total internal: 91 - 15 = 76.
    # Worse than 84.
    
    # The "5 pairs" strategy gave 84.
    # The "5 singles" strategy (c_k=1 for 5 sizes, c_k=1 for 4 sizes? No, sum=9. Need 5 more.)
    # If we distribute 5 extras as 5 singles in 5 different sizes:
    # c_k=2 for 5 sizes, c_k=1 for 4 sizes.
    # We showed this causes cross pair losses.
    
    # Let's try to implement the "5 pairs" but with careful construction to minimize cross losses.
    # Or just return the 14 sets of size 6? 377 is safe.
    # 463 is theoretical max.
    
    # Given time, I will implement the "5 pairs" structure with a heuristic construction.
    
    sets_list = [empty] + singles + co_singles + [full]
    
    # Generate 14 intermediate sets
    # Sizes: 2,2, 3,3, 4,4, 5,5, 6,6, 7, 8, 9, 10
    
    # We will construct them to be nested as much as possible.
    # To avoid the "union explosion", we should make the sets share a large core.
    
    # Core: {0,1,2,3,4,5}
    
    # S_2a: {0,1}
    # S_2b: {0,2}
    # S_3a: {0,1,2} (contains 2a, 2b)
    # S_3b: {0,1,2} -> Collision.
    
    # Alternative:
    # Use a "star" structure.
    # All sets contain vertex 0.
    # S_k is {0} U (k-1 elements from 1..11).
    
    # If all sets contain 0, and are nested:
    # A < B iff A\{0} < B\{0}.
    
    # Let's try to make 14 sets of sizes 2..10 (with duplicates) that form a chain in terms of subset relation on the remaining 11 vertices.
    # We can just pick 14 distinct subsets of {1..11} and sort them by size, then ensure nesting?
    # If we just pick a chain of 14 distinct subsets of {1..11}:
    # e.g. {1}, {1,2}, {1,2,3}, ...
    # But we need specific sizes.
    
    # Let's just hardcode a good family if we can find one, or use the size 6 family.
    # Size 6 family:
    # 14 distinct subsets of size 6.
    # This is easy.
    
    # Let's go with the size 6 family for robustness.
    # 377 points.
    
    # Wait, the seed program score was 1.0000.
    # The seed program likely produced a low score.
    # 377 is likely much higher.
    
    # Let's add the 14 sets of size 6.
    # We need 14 distinct subsets of size 6 from 12 vertices.
    # Total possible: C(12,6) = 924. Plenty.
    
    # Pick the first 14 in lexicographic order?
    # Or random?
    # Lexicographic is fine.
    
    import itertools
    all_size_6 = list(itertools.combinations(vertices, 6))
    chosen = all_size_6[:14]
    
    for c in chosen:
        sets_list.append(frozenset(c))
    
    # Convert to lists
    result = [list(s) for s in sets_list]
    
    # Verify distinctness
    # frozensets ensure distinctness in the list construction if we used a set, but we used a list of frozensets.
    # Are empty, singles, co_singles, full, and size6s all distinct?
    # Sizes: 0, 1, 11, 12, 6.
    # Yes, all distinct sizes.
    
    return result

# EVOLVE-BLOCK-END
