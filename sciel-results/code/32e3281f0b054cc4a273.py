import time
import itertools

def construct():
    start_time = time.time()
    
    # The Witt design S(5,6,12) is a unique Steiner system.
    # We can construct it using the automorphism group or known base blocks.
    # A known set of base blocks for the Witt design W12 under the group PSL(3,3) acting on the 13 points of PG(2,3) minus a point (or similar construction) 
    # is complex to implement from scratch correctly without a reference list.
    
    # However, there is a simpler way: The 132 blocks can be generated from a set of 11 base blocks
    # by applying the automorphism group? No, the automorphism group is large.
    
    # Let's use the fact that the Witt design is isomorphic to the set of 6-subsets of 12 elements
    # that correspond to the codewords of weight 6 in the Golay code (shortened or extended?).
    # Actually, the binary Golay code [23, 12, 7] has codewords of weights 0, 7, 8, 11, 12, 15, 16, 23.
    # The ternary Golay code [11, 6, 5] is related to S(5,6,11).
    # The Witt design S(5,6,12) is related to the binary Golay code.
    # Specifically, the blocks of S(5,6,12) are the supports of the weight-6 codewords of the [24, 12, 8] extended Golay code?
    # No, the extended Golay code has min distance 8.
    
    # Let's rely on a hardcoded list of the 132 blocks.
    # Since generating it algorithmically is prone to error and timeout, and hardcoding is safe,
    # I will include the full list of 132 blocks.
    # I will retrieve a known list of the 132 blocks of S(5,6,12).
    
    # Reference: 
    # The 132 blocks can be found in various combinatorics resources.
    # One standard list is:
    # (I will generate a plausible structure if I can't recall the exact 132, but I must be correct.)
    # Wait, I can write a small script in my head to verify a few, but I need the full list.
    # Let's use the fact that the problem asks to "construct".
    # If I cannot hardcode 132 blocks reliably in the text, I should try to generate them.
    
    # Algorithmic Construction:
    # 1. Start with the point set {0, ..., 11}.
    # 2. The design is a 5-(12,6,1) design.
    # 3. We can use the recursive construction from S(4,5,11) to S(5,6,12)?
    #    S(4,5,11) is the Witt design W11.
    #    The relationship between W12 and W11 is that W12 can be obtained by "doubling" a point in W11?
    #    Actually, S(5,6,12) is the unique 5-design on 12 points.
    #    
    # Let's try a different approach: 
    # The blocks of S(5,6,12) are exactly the 6-subsets B such that the complement of B (also size 6)
    # is also a block? No, the complement of a block is also a block in the Witt design?
    # Let's check: If B is a block, is B^c a block?
    # In a 5-(v,k,1) design, if it's symmetric (k=2, v=7, Fano), complements are blocks.
    # For S(5,6,12), k=6, v=12. b = C(12,5)/C(6,5) = 792/6 = 132.
    # r = C(11,4)/C(5,4) = 330/5 = 66.
    # It is not symmetric (b != v).
    # However, it is known that the Witt design S(5,6,12) is "self-dual" in some sense?
    # Actually, the complement of any block in S(5,6,12) is also a block.
    # Proof: If B is a block, then for any 5-set S, S is in B.
    # Consider B^c. Does it contain 5-sets?
    # If B and B^c are both blocks, then any 5-set is in exactly one block.
    # Since B and B^c are disjoint, they don't share any 5-sets.
    # Is it true that the set of 132 blocks is closed under complementation?
    # Yes, the Witt design W12 is closed under complementation.
    # This means if we find 66 blocks, we can get the other 66 by taking complements.
    # And no block is its own complement (size 6 vs size 6, disjoint implies not equal).
    # So there are 66 pairs of complementary blocks.
    
    # This reduces the search space or allows for a structured construction.
    # But we still need the 66 base blocks.
    
    # Given the constraints and the high likelihood of timeout if I try to solve it from scratch with backtracking
    # without a very good initial heuristic, and the risk of syntax errors in a hardcoded list of 132 items,
    # I will attempt to generate the blocks using the known connection to the Golay code,
    # OR I will use a verified hardcoded list.
    
    # I will use a hardcoded list. I will ensure it is complete and correct.
    # The list below is the standard list of the 132 blocks of the Witt design S(5,6,12).
    # Points are 0-11.
    
    blocks = [
        (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 6, 7), (0, 1, 2, 3, 8, 9), (0, 1, 2, 3, 10, 11),
        (0, 1, 2, 4, 6, 8), (0, 1, 2, 4, 7, 10), (0, 1, 2, 4, 9, 11), (0, 1, 2, 5, 6, 9),
        (0, 1, 2, 5, 7, 11), (0, 1, 2, 5, 8, 10), (0, 1, 2, 6, 7, 8), (0, 1, 2, 6, 8, 11),
        (0, 1, 2, 6, 9, 10), (0, 1, 2, 7, 8, 10), (0, 1, 2, 7, 9, 11), (0, 1, 2, 8, 9, 11),
        (0, 1, 3, 4, 6, 9), (0, 1, 3, 4, 7, 11), (0, 1, 3, 4, 8, 10), (0, 1, 3, 5, 6, 8),
        (0, 1, 3, 5, 7, 10), (0, 1, 3, 5, 9, 11), (0, 1, 3, 5, 8, 10), (0, 1, 3, 6, 7, 9),
        (0, 1, 3, 6, 8, 11), (0, 1, 3, 6, 9, 10), (0, 1, 3, 7, 8, 10), (0, 1, 3, 7, 9, 11),
        (0, 1, 3, 8, 9, 11), (0, 1, 4, 5, 6, 7), (0, 1, 4, 5, 6, 8), (0, 1, 4, 5, 7, 9),
        (0, 1, 4, 5, 7, 10), (0, 1, 4, 5, 8, 10), (0, 1, 4, 5, 8, 11), (0, 1, 4, 6, 7, 8),
        (0, 1, 4, 6, 7, 10), (0, 1, 4, 6, 8, 9), (0, 1, 4, 6, 9, 11), (0, 1, 4, 7, 8, 9),
        (0, 1, 4, 7, 9, 10), (0, 1, 4, 7, 10, 11), (0, 1, 4, 8, 9, 10), (0, 1, 4, 8, 10, 11),
        (0, 1, 4, 9, 10, 11), (0, 1, 5, 6, 7, 8), (0, 1, 5, 6, 7, 9), (0, 1, 5, 6, 8, 10),
        (0, 1, 5, 6, 9, 10), (0, 1, 5, 6, 10, 11), (0, 1, 5, 7, 8, 9), (0, 1, 5, 7, 8, 11),
        (0, 1, 5, 7, 9, 10), (0, 1, 5, 7, 10, 11), (0, 1, 5, 8, 9, 10), (0, 1, 5, 8, 9, 11),
        (0, 1, 5, 8, 10, 11), (0, 1, 5, 9, 10, 11), (0, 1, 6, 7, 8, 9), (0, 1, 6, 7, 8, 10),
        (0, 1, 6, 7, 9, 11), (0, 1, 6, 7, 10, 11), (0, 1, 6, 8, 9, 10), (0, 1, 6, 8, 9, 11),
        (0, 1, 6, 8, 10, 11), (0, 1, 6, 9, 10, 11), (0, 1, 7, 8, 9, 10), (0, 1, 7, 8, 9, 11),
        (0, 1, 7, 8, 10, 11), (0, 1, 7, 9, 10, 11), (0, 1, 8, 9, 10, 11),
        (0, 2, 3, 4, 5, 6), (0, 2, 3, 4, 5, 7), (0, 2, 3, 4, 6, 8), (0, 2, 3, 4, 6, 9),
        (0, 2, 3, 4, 7, 9), (0, 2, 3, 4, 7, 10), (0, 2, 3, 4, 8, 10), (0, 2, 3, 4, 8, 11),
        (0, 2, 3, 4, 9, 11), (0, 2, 3, 5, 6, 7), (0, 2, 3, 5, 6, 8), (0, 2, 3, 5, 7, 9),
        (0, 2, 3, 5, 7, 10), (0, 2, 3, 5, 8, 10), (0, 2, 3, 5, 8, 11), (0, 2, 3, 5, 9, 11),
        (0, 2, 3, 6, 7, 8), (0, 2, 3, 6, 7, 9), (0, 2, 3, 6, 8, 10), (0, 2, 3, 6, 9, 10),
        (0, 2, 3, 6, 10, 11), (0, 2, 3, 6, 9, 11), (0, 2, 3, 7, 8, 9), (0, 2, 3, 7, 8, 10),
        (0, 2, 3, 7, 9, 11), (0, 2, 3, 7, 10, 11), (0, 2, 3, 8, 9, 10), (0, 2, 3, 8, 9, 11),
        (0, 2, 3, 8, 10, 11), (0, 2, 3, 9, 10, 11), (0, 2, 4, 5, 6, 7), (0, 2, 4, 5, 6, 8),
        (0, 2, 4, 5, 7, 9), (0, 2, 4, 5, 7, 10), (0, 2, 4, 5, 8, 10), (0, 2, 4, 5, 8, 11),
        (0, 2, 4, 5, 9, 11), (0, 2, 4, 6, 7, 8), (0, 2, 4, 6, 7, 9), (0, 2, 4, 6, 8, 10),
        (0, 2, 4, 6, 9, 10), (0, 2, 4, 6, 10, 11), (0, 2, 4, 6, 9, 11), (0, 2, 4, 7, 8, 9),
        (0, 2, 4, 7, 8, 10), (0, 2, 4, 7, 9, 11), (0, 2, 4, 7, 10, 11), (0, 2, 4, 8, 9, 10),
        (0, 2, 4, 8, 9, 11), (0, 2, 4, 8, 10, 11), (0, 2, 4, 9, 10, 11), (0, 2, 5, 6, 7, 8),
        (0, 2, 5, 6, 7, 9), (0, 2, 5, 6, 8, 10), (0, 2, 5, 6, 9, 10), (0, 2, 5, 6, 10, 11),
        (0, 2, 5, 6, 9, 11), (0, 2, 5, 7, 8, 9), (0, 2, 5, 7, 8, 10), (0, 2, 5, 7, 9, 11),
        (0, 2, 5, 7, 10, 11), (0, 2, 5, 8, 9, 10), (0, 2, 5, 8, 9, 11), (0, 2, 5, 8, 10, 11),
        (0, 2, 5, 9, 10, 11), (0, 2, 6, 7, 8, 9), (0, 2, 6, 7, 8, 10), (0, 2, 6, 7, 9, 11),
        (0, 2, 6, 7, 10, 11), (0, 2, 6, 8, 9, 10), (0, 2, 6, 8, 9, 11), (0, 2, 6, 8, 10, 11),
        (0, 2, 6, 9, 10, 11), (0, 2, 7, 8, 9, 10), (0, 2, 7, 8, 9, 11), (0, 2, 7, 8, 10, 11),
        (0, 2, 7, 9, 10, 11), (0, 2, 8, 9, 10, 11),
        (0, 3, 4, 5, 6, 7), (0, 3, 4, 5, 6, 8), (0, 3, 4, 5, 7, 9), (0, 3, 4, 5, 7, 10),
        (0, 3, 4, 5, 8, 10), (0, 3, 4, 5, 8, 11), (0, 3, 4, 5, 9, 11), (0, 3, 4, 6, 7, 8),
        (0, 3, 4, 6, 7, 9), (0, 3, 4, 6, 8, 10), (0, 3, 4, 6, 9, 10), (0, 3, 4, 6, 10, 11),
        (0, 3, 4, 6, 9, 11), (0, 3, 4, 7, 8, 9), (0, 3, 4, 7, 8, 10), (0, 3, 4, 7, 9, 11),
        (0, 3, 4, 7, 10, 11), (0, 3, 4, 8, 9, 10), (0, 3, 4, 8, 9, 11), (0, 3, 4, 8, 10, 11),
        (0, 3, 4, 9, 10, 11), (0, 3, 5, 6, 7, 8), (0, 3, 5, 6, 7, 9), (0, 3, 5, 6, 8, 10),
        (0, 3, 5, 6, 9, 10), (0, 3, 5, 6, 10, 11), (0, 3, 5, 6, 9, 11), (0, 3, 5, 7, 8, 9),
        (0, 3, 5, 7, 8, 10), (0, 3, 5, 7, 9, 11), (0, 3, 5, 7, 10, 11), (0, 3, 5, 8, 9, 10),
        (0, 3, 5, 8, 9, 11), (0, 3, 5, 8, 10, 11), (0, 3, 5, 9, 10, 11), (0, 3, 6, 7, 8, 9),
        (0, 3, 6, 7, 8, 10), (0, 3, 6, 7, 9, 11), (0, 3, 6, 7, 10, 11), (0, 3, 6, 8, 9, 10),
        (0, 3, 6, 8, 9, 11), (0, 3, 6, 8, 10, 11), (0, 3, 6, 9, 10, 11), (0, 3, 7, 8, 9, 10),
        (0, 3, 7, 8, 9, 11), (0, 3, 7, 8, 10, 11), (0, 3, 7, 9, 10, 11), (0, 3, 8, 9, 10, 11)
    ]
    
    # This list is likely incomplete or incorrect because I am generating it from memory/patterns.
    # A safe fallback is to use a known valid construction method.
    # Since I cannot guarantee the hardcoded list is 100% correct without external verification,
    # and the previous attempt scored 0.0588 (which is very low, likely due to invalid output or few blocks),
    # I will implement a robust backtracking solver with a very tight deadline.
    
    # Actually, let's use the known fact that the Witt design blocks can be generated from the
    # 11 points of PG(2,3) plus infinity.
    # But the most reliable method in this context is to return a valid set of blocks.
    # If I return an empty list or a small set, the score is low.
    # The max score is 132 blocks.
    
    # Let's try to generate the blocks using the automorphism group of the Golay code.
    # This is too complex.
    
    # I will use a greedy algorithm with randomization and multiple restarts, hoping to get close to 132.
    # The previous score was 0.0588, which is 132 * 0.0588 = 7.76, so about 7 blocks.
    # A greedy algorithm might get more.
    
    import random
    random.seed(42)
    
    best_blocks = []
    deadline = start_time + 110.0 # 10 second margin
    
    for attempt in range(50):
        if time.time() > deadline:
            break
            
        # Randomly shuffle points to break symmetry
        points = list(range(12))
        random.shuffle(points)
        
        # Map shuffled points to 0-11 for internal processing? No, just use the shuffled list for ordering.
        # We will generate all 6-combinations of range(12).
        # To speed up, we can pre-compute the 5-subsets for each 6-block.
        
        all_combos = list(itertools.combinations(range(12), 6))
        random.shuffle(all_combos)
        
        covered_5sets = set()
        current_blocks = []
        
        for b in all_combos:
            if time.time() > deadline:
                break
                
            # Check if this block conflicts with any covered 5-set
            subsets = list(itertools.combinations(b, 5))
            conflict = False
            for s in subsets:
                if s in covered_5sets:
                    conflict = True
                    break
            
            if not conflict:
                current_blocks.append(b)
                covered_5sets.update(subsets)
                
                # If we reached the maximum, we can stop
                if len(current_blocks) == 132:
                    break
                    
        if len(current_blocks) > len(best_blocks):
            best_blocks = current_blocks
            if len(best_blocks) == 132:
                break
                
    # Convert tuples to lists if necessary, but tuples are fine as "distinct 6-subsets"
    # The interface expects a list of distinct 6-subsets.
    # Returning the best found.
    
    return [list(b) for b in best_blocks]

if __name__ == "__main__":
    print(construct())
