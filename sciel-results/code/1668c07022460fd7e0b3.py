import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 55  # Safety margin
    
    best_set = [31, 32, 34]
    best_size = 3
    
    def construct_binary_signature():
        """
        Use binary representation approach: assign each integer a unique power of 2 as its "signature".
        This ensures all subset sums are distinct. Then filter for divisibility constraints.
        """
        # Strategy: Select numbers that have distinct "dominant" bits
        # A number with a unique high bit won't create collisions in subset sums
        # We want numbers where each has a different highest set bit
        
        # Group numbers 1..256 by their highest set bit
        by_highest_bit = {}
        for n in range(1, 257):
            hb = n.bit_length() - 1  # 0-indexed position of highest bit
            if hb not in by_highest_bit:
                by_highest_bit[hb] = []
            by_highest_bit[hb].append(n)
        
        # For each bit position, we can pick at most one number
        # to ensure subset sum uniqueness (each subset has a unique set of highest bits)
        
        # Try different strategies for picking which number from each bit group
        best_local = []
        
        # Strategy 1: Pick the largest number from each bit group (maximizes values)
        candidate_sets = []
        
        # Greedy: try to build a set by picking from each bit group
        # We'll try multiple orderings
        
        for attempt in range(20):
            if time.time() - start_time > time_limit:
                break
                
            # Create a list of (bit_pos, candidate_number) pairs
            # Shuffle the bit positions to try different combinations
            bit_positions = sorted(by_highest_bit.keys(), reverse=True)
            random.shuffle(bit_positions)
            
            S = []
            sums = {0}
            
            for bp in bit_positions:
                if time.time() - start_time > time_limit:
                    break
                    
                candidates = by_highest_bit[bp]
                # Try candidates in random order
                random.shuffle(candidates)
                
                for x in candidates:
                    # Check if adding x keeps the set valid
                    new_sums = set()
                    for s in sums:
                        new_sums.add(s + x)
                    
                    # Check distinctness: new_sums should not overlap with existing sums
                    # except for 0
                    if new_sums & sums:
                        continue
                    
                    # Check divisibility
                    non_empty_new = [s for s in new_sums if s != 0]
                    non_empty_old = [s for s in sums if s != 0]
                    
                    valid = True
                    # Check internal new
                    for i in range(len(non_empty_new)):
                        u = non_empty_new[i]
                        for j in range(len(non_empty_new)):
                            if i == j:
                                continue
                            v = non_empty_new[j]
                            if v % u == 0 or u % v == 0:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        continue
                    
                    # Check cross
                    for u in non_empty_new:
                        for v in non_empty_old:
                            if v % u == 0 or u % v == 0:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        continue
                    
                    # Valid! Add x
                    S.append(x)
                    sums = sums | new_sums
                    break
            
            if len(S) > len(best_local):
                best_local = S[:]
        
        return best_local
    
    def is_valid(S):
        if not S:
            return True
        
        # Generate all non-empty subset sums
        # Use list for efficient iteration, then set for uniqueness check
        sums = [0]
        for x in S:
            # Add x to all existing sums
            # This creates new sums
            new_sums = [s + x for s in sums]
            sums.extend(new_sums)
        
        # Remove 0
        non_empty_sums = [s for s in sums if s != 0]
        
        # Check distinctness
        if len(set(non_empty_sums)) != len(non_empty_sums):
            return False
        
        # Check divisibility
        # Sort to potentially optimize? Not really needed for O(N^2)
        n = len(non_empty_sums)
        for i in range(n):
            u = non_empty_sums[i]
            for j in range(n):
                if i == j:
                    continue
                v = non_empty_sums[j]
                # If u divides v or v divides u
                # Optimization: only check if v > u? 
                # If u > v, then v divides u is the check.
                # We can just check both.
                if v % u == 0 or u % v == 0:
                    return False
        return True

    def try_extend(S, candidates):
        """Tries to add one element from candidates to S."""
        for x in candidates:
            if x in S:
                continue
            S_new = S + [x]
            if is_valid(S_new):
                return S_new
        return None

    # Strategy:
    # 1. Try to find a large set using random initialization and greedy addition.
    # 2. If greedy gets stuck, use local search.
    
    # To make it fast, we pre-filter candidates.
    # Small numbers are bad because they divide many things.
    # Let's prioritize numbers > 50.
    
    # We will run multiple random restarts.
    # For each restart, we build a set greedily.
    
    # To improve search, we can use a "pool" of good candidates.
    # But for now, let's stick to randomized greedy with restarts.
    
    # We can also try to start from the best known set and perturb it.
    
    # Initial pool: all numbers 1..256
    # But we can bias towards larger numbers.
    
    # Let's run a loop.
    # We will maintain a list of "frontier" sets.
    # But simple restarts might be enough if we are smart about ordering.
    
    # Better: Start with empty set.
    # Add random elements.
    # If valid, keep.
    # If not, discard that element.
    # This is a simple greedy. It gets stuck easily.
    
    # Improved: 
    # 1. Generate a random permutation of 1..256.
    # 2. Iterate through it.
    # 3. If adding the current element to the current set is valid, add it.
    # 4. This yields a maximal set (with respect to the order).
    # 5. Do this many times with different random seeds/orders.
    
    # Additionally, we can try to "swap" elements in a found set to try to grow it.
    # But swapping reduces size by 1, then we try to add 2?
    
    # Let's just do many random greedy attempts.
    
    # We can speed up is_valid by checking the most likely failures first.
    # Small sums are more likely to divide large sums.
    
    # Optimization: 
    # If S is valid, and we add x, the new sums are old_sums and old_sums + x.
    # We only need to check:
    # 1. Distinctness of new sums vs old sums.
    # 2. Divisibility between new sums and old sums.
    # 3. Divisibility within new sums.
    
    def is_valid_incremental(S, x):
        # S is valid.
        # Generate sums of S
        sums_S = {0}
        for val in S:
            new_sums = set()
            for s in sums_S:
                new_sums.add(s + val)
            sums_S = sums_S | new_sums
        
        # New sums are x + s for s in sums_S
        new_sums = {x + s for s in sums_S}
        
        # Check distinctness
        # Intersection of sums_S and new_sums must be empty
        if sums_S & new_sums:
            return False
            
        # Check divisibility
        # All pairs in (sums_S U new_sums) \ {0}
        # We know sums_S \ {0} is antichain.
        # We need to check:
        # - new_sums \ {0} is antichain
        # - cross pairs
        
        non_empty_new = [s for s in new_sums if s != 0]
        non_empty_old = [s for s in sums_S if s != 0]
        
        # Check internal new
        n = len(non_empty_new)
        for i in range(n):
            u = non_empty_new[i]
            for j in range(n):
                if i == j: continue
                v = non_empty_new[j]
                if v % u == 0 or u % v == 0:
                    return False
        
        # Check cross
        for u in non_empty_new:
            for v in non_empty_old:
                if v % u == 0 or u % v == 0:
                    return False
                    
        return True

    # First, try the binary signature approach
    binary_result = construct_binary_signature()
    if len(binary_result) > best_size:
        best_size = len(binary_result)
        best_set = binary_result[:]
    
    # Then continue with randomized greedy search
    # We will use the incremental check to speed up the greedy building.
    
    max_restarts = 10000
    # We can also vary the range of numbers.
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Randomize the order of candidates
        # Bias towards larger numbers?
        # Let's try uniform first.
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        S = []
        # Greedy build
        for x in candidates:
            if len(S) >= best_size + 2:
                # If we are already way ahead, we can be more selective?
                # No, just keep going.
                pass
            
            # Only try to add if it doesn't exceed time
            if time.time() - start_time > time_limit - 1:
                break
                
            if is_valid_incremental(S, x):
                S.append(x)
                
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S[:]
                    
            # If S becomes large, incremental check is slower.
            # 4^k complexity.
            # k=10 -> 1M.
            # k=12 -> 16M.
            # If we reach 10, we are doing well.
            # We might want to break early if S is very large to save time for other restarts?
            # No, we want to maximize size.
            
            if len(S) > 12:
                # Check is very slow.
                # If we are here, we likely found a good set.
                # Let's do a few more tries?
                # Or break to start a new restart?
                # Actually, if we break, we might miss a larger set.
                # But 4^13 is 67M. ~0.5s per check.
                # We can only do ~100 checks in 50s.
                # So if we are at k=12, we should be careful.
                break

    # Final verification
    if len(best_set) <= 12:
        if not is_valid(best_set):
             # Fallback
             best_set = [31, 32, 34]
    else:
        # Trust the incremental check, but it's risky if bugs exist.
        # For large sets, we assume incremental logic is correct.
        pass

    return list(best_set)

if __name__ == "__main__":
    res = construct()
    print(res)
