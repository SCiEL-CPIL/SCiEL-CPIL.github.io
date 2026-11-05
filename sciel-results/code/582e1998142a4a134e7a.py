import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    # The problem reduces to finding the largest subset of [1, 100] with no subset summing to 200.
    # Since the max sum of 2 distinct elements is 199, we only need to worry about subsets of size >= 3.
    # If we restrict the universe to U = {51, ..., 100}, the min sum of 4 distinct elements is 51+52+53+54=210 > 200.
    # So for U, we only need to avoid 3-element subsets summing to 200.
    # This simplifies the check significantly.
    
    U = set(range(51, 101))
    
    def get_violations(S):
        """Returns a list of tuples (x, y, z) in S such that x+y+z=200, x<y<z."""
        S_list = sorted(S)
        n = len(S_list)
        violations = []
        # To find 3-sums, we can iterate pairs and check if the third is in S.
        # But we need distinct elements.
        # Efficient way: for each pair (i, j) with i < j, check if 200 - S[i] - S[j] is in S and > S[j].
        S_set = set(S)
        for i in range(n):
            for j in range(i + 1, n):
                rem = 200 - S_list[i] - S_list[j]
                if rem > S_list[j] and rem in S_set:
                    violations.append((S_list[i], S_list[j], rem))
        return violations

    def is_valid(S):
        return len(get_violations(S)) == 0

    def greedy_solve(random_seed):
        random.seed(random_seed)
        S = set(U)
        
        # Greedy removal
        while True:
            if time.time() > deadline:
                break
            viols = get_violations(S)
            if not viols:
                break
            
            # Count frequency of each element in violations
            freq = {}
            for x, y, z in viols:
                freq[x] = freq.get(x, 0) + 1
                freq[y] = freq.get(y, 0) + 1
                freq[z] = freq.get(z, 0) + 1
            
            # Remove the element with the highest frequency
            # To add randomness, pick from the top 3 if tied or close
            max_freq = max(freq.values())
            candidates = [k for k, v in freq.items() if v == max_freq]
            # Add some randomness: pick randomly from top candidates
            elem_to_remove = random.choice(candidates)
            S.remove(elem_to_remove)
        
        # Local search: try to add elements back
        # Sort candidates by value? Or random order?
        # Elements that are "less connected" might be easier to add.
        # Let's just iterate through all missing elements.
        missing = [x for x in U if x not in S]
        random.shuffle(missing)
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for x in missing:
                if x in S:
                    continue
                new_S = S | {x}
                if is_valid(new_S):
                    S.add(x)
                    changed = True
                    # Update missing list is expensive, just check membership
        return S

    best_S = set()
    
    # Run multiple restarts with different seeds
    # The greedy removal is deterministic-ish but we randomize the tie-breaking.
    # We can also randomize the initial set slightly? 
    # No, starting with full U is a good upper bound.
    
    num_restarts = 100
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        S = greedy_solve(i)
        if len(S) > len(best_S):
            best_S = S
            # If we found a very good solution, we can stop early?
            # Max possible is 50.
            if len(best_S) >= 45:
                break

    # Final polish: Try to add any remaining elements from [1, 100] not in U?
    # Adding elements < 51 creates 4-sum risks.
    # If we have a valid S in U, can we add x < 51?
    # We need to check if there exist y, z, w in S such that x+y+z+w=200.
    # This is a 3-sum problem in S for target 200-x.
    
    # Let's try to add elements from 1 to 50 to best_S
    current_S = set(best_S)
    candidates_small = list(range(1, 51))
    random.shuffle(candidates_small)
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in candidates_small:
            if x in current_S:
                continue
            # Check if adding x creates a sum of 200
            # 1. x + y + z + w = 200 (4 distinct from current_S)
            # 2. x + y + z = 200 (3 distinct from current_S, one is x) -> y+z = 200-x
            # 3. x + y = 200 (2 distinct) -> y = 200-x. But y<=100, x>=1 => y<=99.
            #    Since current_S subset of [51, 100], y must be in [51, 99].
            #    If 200-x is in current_S, then {x, 200-x} is a 2-sum.
            #    Wait, the problem says "no subset summing to 200".
            #    So {x, 200-x} is forbidden.
            
            valid = True
            target_2 = 200 - x
            if target_2 in current_S:
                valid = False
            else:
                # Check 3-sum: x + y + z = 200 => y + z = 200 - x
                # y, z in current_S, y != z
                # Check 4-sum: x + y + z + w = 200 => y + z + w = 200 - x
                # y, z, w in current_S, distinct
                
                if valid:
                    # Check 3-sums involving x
                    # y+z = 200-x
                    # Iterate pairs in current_S?
                    # current_S size is ~40. Pairs ~800.
                    # We can check if 200-x is a subset sum of current_S of size 2.
                    # Since current_S elements are >= 51, min 2-sum is 102.
                    # 200-x for x in 1..50 is 150..199.
                    # So 2-sums are possible.
                    found_2_sum = False
                    S_list = list(current_S)
                    for i in range(len(S_list)):
                        for j in range(i+1, len(S_list)):
                            if S_list[i] + S_list[j] == 200 - x:
                                found_2_sum = True
                                break
                        if found_2_sum:
                            break
                    if found_2_sum:
                        valid = False
                
                if valid:
                    # Check 4-sums involving x
                    # y+z+w = 200-x
                    # y,z,w in current_S, distinct.
                    # Min 3-sum in current_S is 51+52+53=156.
                    # 200-x for x in 1..50 is 150..199.
                    # So 3-sums are possible if 200-x >= 156 => x <= 44.
                    if x <= 44:
                        found_3_sum = False
                        # Check if 200-x is a subset sum of current_S of size 3.
                        # This is more expensive.
                        # We can use a set of current_S.
                        S_set = set(current_S)
                        S_list = list(current_S)
                        # Optimize: iterate pairs, check if rem is in S_set and distinct.
                        for i in range(len(S_list)):
                            for j in range(i+1, len(S_list)):
                                rem = (200 - x) - S_list[i] - S_list[j]
                                if rem > S_list[j] and rem in S_set:
                                    found_3_sum = True
                                    break
                            if found_3_sum:
                                break
                        if found_3_sum:
                            valid = False
            
            if valid:
                current_S.add(x)
                changed = True
    
    return list(current_S)

# To ensure we don't time out in the inner loops of the final polish, 
# we rely on the deadline check in the while loop.
# The inner loops are bounded by size of S (~50), so ~2500 iterations max per x.
# 50 x's * 2500 = 125,000 ops. Very fast.

if __name__ == "__main__":
    res = construct()
    print(len(res))
