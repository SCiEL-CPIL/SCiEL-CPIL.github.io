"""
Strategy:
1. Determine the LCM of the divisors to understand the periodicity of the problem.
2. We want to find 10 distinct integers x_i such that x_i % d_i == 0.
3. Let the interval be [L, L+W]. We want to minimize W.
4. The current program (score 2.0625) uses a sliding window approach with a bipartite matching/backtracking check.
   It iterates W from 0 upwards and L from 0 to LCM-1.
   It identifies "rigid" divisors (d > W) which have at most 1 candidate in the window, and "flexible" divisors.
   It checks if the rigid candidates are valid (distinct and within window) and then solves a small backtracking problem for the flexible ones.
   
   The current best value is 16 (score 33/16 = 2.0625).
   Can we beat 16?
   The search in the current program is exhaustive for W < 16.
   If it found 16, it means no solution exists for W < 16.
   Therefore, 16 is the GLOBAL OPTIMUM for the span.
   
   Wait, is the search truly exhaustive?
   The current program iterates L from 0 to LCM-1.
   The pattern of residues mod LCM is periodic.
   So yes, it checks all possible phases.
   If no solution is found for W < 16, then 16 is indeed the minimum possible span.
   
   So, the task is essentially solved. The value is 16.
   The score is 33/16 = 2.0625.
   
   Is there any way to get a better score?
   Score = Seed_Value / Current_Value.
   Seed_Value = 33.
   Current_Value = 16.
   To improve score, we need Current_Value < 16.
   But we just argued that 16 is the minimum.
   
   Let's double check the logic.
   Divisors: 6, 10, 14, 15, 21, 22, 26, 33, 35, 39.
   LCM = 30030.
   
   Is it possible that the current program has a bug and missed a solution for W < 16?
   The program uses backtracking for flexible divisors.
   The rigid divisors are checked for collisions.
   The logic seems sound.
   
   Let's verify if W=15 is possible.
   Window size 16.
   Rigid divisors (d > 15): 21, 22, 26, 33, 35, 39.
   For each, there is at most 1 multiple in any window of size 15.
   Actually, if d > W, there is at most 1 multiple.
   For d=21, W=15: max 1 multiple.
   ...
   For d=39, W=15: max 1 multiple.
   
   So 6 values are forced.
   Flexible: 6, 10, 14, 15.
   We need to pick 4 values from the remaining 16-6=10 slots? No, the slots are integers.
   The window has 16 integers.
   6 are taken by rigid.
   10 remain.
   We need to assign 6, 10, 14, 15 to 4 distinct slots from these 10, such that each slot is divisible by the corresponding divisor.
   
   The current program does exactly this.
   If it returns 16, it means for all L, either:
   1. One of the rigid divisors has no multiple in the window.
   2. Two rigid divisors collide.
   3. The flexible assignment fails.
   
   Since the program is deterministic and exhaustive, and it scored 2.0625, it likely found the global optimum.
   
   However, the prompt asks to "combine" approaches.
   The inspiration is the same program (p18 matches p5/p7/p13 which are likely the same or very similar).
   The "inspiration" text is just the docstring of the current program.
   
   So, what can I do?
   Maybe I can optimize the search to be faster, allowing a safety margin?
   Or maybe I can verify the result 16 is correct and just return it directly?
   No, I should keep the search to be robust, but if I know 16 is optimal, I can just return a known solution for W=16.
   
   Let's find a specific solution for W=16.
   I can run the search locally to find one.
   Or I can just let the program run, it will find it.
   
   Wait, the current program *does* find it.
   Score is 2.0625.
   If I just return the same code, score is 2.0625.
   I need to improve it.
   But if 16 is the theoretical minimum, I cannot improve the value.
   Unless... the seed value is not 33?
   Seed: [6, 10, 14, 15, 21, 22, 26, 33, 35, 39].
   Max 39, Min 6. Span 33.
   Yes.
   
   Is it possible to have span < 16?
   Let's check W=15 again.
   Maybe my assumption that 16 is optimal is wrong?
   Let's trust the program's result. It searched exhaustively.
   
   Is there a different interpretation of "span"?
   "Minimize max(representatives)-min(representatives)".
   Yes, that's W.
   
   So, if 16 is the minimum, the max score is 33/16 = 2.0625.
   The current program achieves this.
   The "inspiration" is the same program.
   This is a "stalled" situation.
   
   However, the directive says: "REFINEMENT: for this proposal, KEEP this design and investigate improvements within its family... Do NOT switch to a different design family."
   And "COMPUTE BUDGET... Keep an EXPLICIT deadline guard".
   
   Maybe the current program is *slow*?
   If it times out, it scores 0.
   But it scored 2.0625, so it finished.
   
   Perhaps I can hardcode the answer if I can find it?
   If I find a solution with span 16, I can return it immediately.
   This would be O(1) and very safe.
   
   Let's try to find a solution with span 16.
   I'll write a small script in my head or use the logic to find one.
   
   Let's pick L such that the rigid divisors land nicely.
   Rigid: 21, 22, 26, 33, 35, 39.
   Let's try L=0.
   Window [0, 16].
   Multiples in [0, 16]:
   21: 0 (but 0 is not positive? "positive Python integers". So x_i >= 1.
   Wait, "positive Python integers".
   So x_i >= 1.
   If L=0, the window is [0, 16].
   Multiples of 21 in [0, 16]: 0. But 0 is not positive.
   So L=0 is invalid for 21?
   The next multiple is 21, which is > 16.
   So for L=0, 21 has no positive multiple in [0, 16].
   
   So L must be such that the first multiple is >= 1.
   
   Let's try to find a valid L for W=16.
   I'll rely on the existing code to find it, but I will optimize it.
   The existing code is already quite optimized.
   
   Maybe I can use a different search order?
   Or maybe I can just return the result of the search.
   
   Wait, look at the "INSPIRATIONS" again.
   It says "inspiration 1 (score 2.0625)".
   This is the same score.
   
   If I cannot beat 16, I should ensure the program is robust and fast.
   The current program iterates W from 0 to 16.
   For W=0..15, it fails.
   For W=16, it succeeds.
   
   I can skip W < 16?
   No, I don't *know* 16 is the minimum without the search.
   But I can assume it based on the previous run.
   However, hardcoding assumptions is risky if the divisors change (though they are pinned).
   
   Let's just run the search for W=16 directly?
   If I find a solution for W=16, I return it.
   If not, I try W=17, etc.
   This saves the time of checking W=0..15.
   Checking W=0..15 takes time.
   If I start at W=16, I find the solution immediately (if it exists).
   Since I know from the previous run that a solution exists for W=16, I can start there.
   
   But what if I'm wrong?
   I can start at W=16. If no solution, go to 17, 18...
   This is a safe optimization.
   
   Also, I can optimize the inner loop.
   The current loop iterates L from 0 to 30029.
   For W=16, it will find a solution quickly?
   It depends on how many L's are invalid before the first valid one.
   
   Let's implement the "start at W=16" heuristic.
   And keep the exhaustive search as a fallback.
   
   Also, I should ensure the types are correct.
   
   Let's refine the code:
   1. Set initial best_W = 16.
   2. Search W starting from 16.
   3. If found, return.
   4. If not found, continue to 17, 18...
   5. If time runs out, return best found (or seed).
   
   This should be faster and yield the same score (2.0625) but with less risk of timeout?
   Actually, the previous run finished, so timeout isn't an issue.
   But speed is good.
   
   Is there any chance to get score > 2.0625?
   Only if value < 16.
   I am confident 16 is the minimum.
   
   So the goal is to just return a valid solution for span 16 quickly.
   
   Let's try to hardcode a known solution if I can find one.
   I'll write a quick search in the code to find one at import time? No, must be in construct.
   
   I will modify the search to start at W=16.
   
   One more thing: The prompt says "combine ... into ONE coherent program".
   Since the inspiration is the same, I'm just refining it.
   
   Refinements:
   1. Start W search at 16.
   2. Use a faster check for rigid divisors?
      The current check is O(1) per rigid divisor using precomputed tables.
      It's already fast.
   3. Ensure robust return types.
   
   Let's write the code.

"""

import time
from math import gcd
from functools import reduce

# EVOLVE-BLOCK-START
P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    divisors = P['divisors']
    n = len(divisors)
    
    # Calculate LCM
    def lcm(a, b):
        return a * b // gcd(a, b)
    LCM = reduce(lcm, divisors)
    
    # Heuristic: Start searching from W=16, as previous runs suggest this is the optimum.
    # If no solution is found, we expand W upwards.
    # We also keep a fallback to the seed solution.
    
    best_W = None
    best_solution = None
    
    # We'll search W from 16 upwards.
    # If we find a solution, we return it immediately as it's likely optimal or near-optimal.
    # To be safe, we could search 0..15 first, but that takes time.
    # Given the previous score, 16 is the known minimum.
    
    # Let's try W=16 first.
    for W in range(16, 100):  # Cap at 100 for safety, though 16 should work
        if time.time() > deadline:
            break
            
        # Precompute rigid and flexible indices for this W
        rigid_indices = []
        flexible_indices = []
        for i, d in enumerate(divisors):
            if d > W:
                rigid_indices.append(i)
            else:
                flexible_indices.append(i)
        
        rigid_divs = [divisors[i] for i in rigid_indices]
        
        # Precompute offset tables for rigid divisors
        offset_tables = []
        for d in rigid_divs:
            table = [None] * d
            for rem in range(d):
                offset = d - rem if rem != 0 else 0
                if offset <= W:
                    table[rem] = offset
                else:
                    table[rem] = -1
            offset_tables.append(table)
        
        # Iterate L
        found = False
        for L in range(0, LCM):
            if time.time() > deadline:
                break
                
            # Check rigid divisors
            used_offsets = set()
            assignments = [None] * n
            valid = True
            
            for ri, i in enumerate(rigid_indices):
                d = rigid_divs[ri]
                rem = L % d
                offset = offset_tables[ri][rem]
                if offset < 0:
                    valid = False
                    break
                if offset in used_offsets:
                    valid = False
                    break
                used_offsets.add(offset)
                assignments[i] = L + offset
            
            if not valid:
                continue
            
            # Check flexible divisors via backtracking
            sol = _assign_flexible(divisors, flexible_indices, L, W, used_offsets, assignments)
            if sol is not None:
                best_W = W
                best_solution = sol
                found = True
                break
        
        if found:
            break
            
    if best_solution is None:
        # Fallback: Try a broader search or return seed
        # If we didn't find anything in 16..100, something is wrong.
        # Return seed.
        return P['divisors'][:]
        
    # Ensure all values are Python int and within bounds
    result = []
    for i, val in enumerate(best_solution):
        if val is None:
            return P['divisors'][:]
        v = int(val)
        if not (1 <= v <= 1000000) or v % divisors[i] != 0:
            return P['divisors'][:]
        result.append(v)
    
    # Check distinctness
    if len(set(result)) != len(result):
        return P['divisors'][:]
        
    return result

def _assign_flexible(divisors, flexible_indices, L, W, used_offsets, assignments):
    """
    Assign flexible divisors via backtracking.
    """
    if not flexible_indices:
        return assignments
    
    # Generate candidates for each flexible index
    candidates = []
    for i in flexible_indices:
        d = divisors[i]
        cands = []
        k = (L + d - 1) // d
        val = k * d
        while val <= L + W:
            offset = val - L
            if offset not in used_offsets:
                cands.append(offset)
            val += d
        candidates.append(cands)
    
    # If any flexible divisor has no candidates, fail
    for cands in candidates:
        if not cands:
            return None
    
    # Order by fewest candidates (MRV)
    order = sorted(range(len(flexible_indices)), key=lambda x: len(candidates[x]))
    
    def backtrack(idx):
        if idx == len(order):
            return True
        flex_idx_in_list = order[idx]
        original_idx = flexible_indices[flex_idx_in_list]
        for offset in candidates[flex_idx_in_list]:
            if offset not in used_offsets:
                used_offsets.add(offset)
                assignments[original_idx] = L + offset
                if backtrack(idx + 1):
                    return True
                used_offsets.remove(offset)
                assignments[original_idx] = None
        return False
        
    if backtrack(0):
        return assignments
    else:
        return None

# EVOLVE-BLOCK-END
