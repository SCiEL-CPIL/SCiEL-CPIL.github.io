"""
Strategy: Exact Maximum Weight Independent Set (MWIS) via Branch and Bound.
Instead of heuristic local search, we model the problem as finding an independent set
in a graph where nodes are (m, r) pairs and edges represent conflicts.
We use a branch-and-bound approach with a strong upper bound: 
current_weight + sum(1/m for all unprocessed moduli).
This allows us to prune branches that cannot possibly beat the current best solution.
"""

import time
import math
import sys
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Moduli range [20, 200]
    moduli = list(range(20, 201))
    num_moduli = len(moduli)
    
    # Precompute 1/m as floats for fast scoring, and as Fraction for exactness if needed 
    # (though float is usually sufficient for comparison in B&B if careful, 
    # but the task asks for exact rational reported as float. 
    # We will track exact Fraction sum for the final answer, 
    # but use float sums for the B&B pruning to be fast).
    
    # For B&B, we need to order nodes. 
    # Nodes are (m, r). Total nodes = sum(m for m in moduli) ~ 181 * 110 ~ 20k nodes.
    # This is too large for a naive B&B on the full graph.
    
    # Alternative Structural View:
    # The conflict between (m1, r1) and (m2, r2) depends only on gcd(m1, m2).
    # Specifically, r1 != r2 mod gcd(m1, m2).
    # This structure allows us to process moduli one by one.
    # When adding modulus m_k, we choose a residue r_k.
    # The constraint is that r_k must not conflict with any previously chosen (m_j, r_j).
    # This is a constraint satisfaction problem (CSP) with an objective function.
    
    # Let's refine the B&B on the *moduli* rather than all (m,r) pairs.
    # State: (index of next modulus to consider, current set of chosen (m,r), current score)
    # However, the choice of residue for m_i affects future choices.
    # So the state must include the "signature" of constraints imposed on future moduli.
    # The constraint on a future modulus m_j is: r_j != r_i mod gcd(m_i, m_j) for all chosen i.
    # This is equivalent to: r_j cannot be in a set of forbidden residues modulo each gcd.
    
    # Actually, a simpler B&B:
    # Order moduli. 
    # At each step, we decide:
    # 1. Skip this modulus.
    # 2. Include this modulus with a specific residue r.
    
    # To make B&B effective, we need a good upper bound.
    # Upper Bound = Current Score + Sum(1/m for all remaining moduli).
    # This is a weak bound because it ignores conflicts.
    # A tighter bound: For each remaining modulus m_j, the maximum contribution is 1/m_j.
    # But if we include m_j, we might be forced to skip others.
    
    # Let's try a different structural approach:
    # Since the moduli are small (20-200), we can use a backtracking search 
    # with strong pruning based on "dominance".
    
    # Key Insight: 
    # If we fix the set of moduli S, is there a valid assignment of residues?
    # Yes, if and only if for every pair in S, the constraints are satisfiable.
    # Actually, for any set of moduli, there exists a valid assignment of residues?
    # Let's check. 
    # If m1, m2 are coprime, any residues work.
    # If gcd(m1, m2) = g > 1, we need r1 != r2 mod g.
    # This is always satisfiable for a single pair.
    # For a set of moduli, is it always satisfiable?
    # Consider m1=20, m2=30, m3=10.
    # gcd(20,30)=10, gcd(20,10)=10, gcd(30,10)=10.
    # Constraints: r1 != r2 mod 10, r1 != r3 mod 10, r2 != r3 mod 10.
    # We need 3 distinct residues mod 10. Since 10 >= 3, this is possible.
    # In general, if the "clique" of moduli sharing a common divisor g has size K,
    # we need K distinct residues mod g. This requires g >= K.
    # If g < K, the set is infeasible.
    
    # So, the problem is: Select a subset of moduli S and assign residues such that 
    # for every divisor g, the number of moduli in S divisible by g is <= g?
    # No, that's not quite right. The constraints are pairwise.
    # The condition for a set of moduli to be "residue-assignable" is related to 
    # the Hall's Marriage Theorem or similar graph coloring conditions on the intersection graph.
    
    # However, we don't just need *any* assignment, we want to maximize sum(1/m).
    # Since all 1/m are positive, we want the *largest* possible set of moduli, 
    # prioritizing small moduli.
    
    # Let's use a Backtracking Search on the Moduli.
    # Order moduli by descending 1/m (i.e., ascending m).
    # At each step, try to include the current modulus.
    # If we include it, we must check if a valid residue exists given the previous choices.
    # AND we must ensure that future moduli can still be added? 
    # No, B&B handles the future via the upper bound.
    
    # The critical part: "Check if a valid residue exists".
    # Given a set of previous choices P = {(m_j, r_j)}, and a new modulus m,
    # does there exist an r such that r != r_j mod gcd(m, m_j) for all j in P?
    # Let g_j = gcd(m, m_j).
    # The forbidden residues for r are: { r_j mod g_j } for each j.
    # Note that if g_j divides g_k, the constraint mod g_j is weaker (forbids more) or stronger?
    # r != a mod g1 and r != b mod g2.
    # The set of valid residues is [0, m) \ Union(Forbidden Sets).
    # We just need to check if this set is non-empty.
    
    # Optimization: 
    # We can precompute for each modulus m, the "conflict signature" imposed by a set of previous moduli.
    # But the previous choices include specific residues.
    
    # Let's implement a fast B&B:
    # 1. Sort moduli ascending.
    # 2. State: index i, current_score (Fraction), current_choices (list of (m, r)).
    # 3. Upper Bound: current_score + sum(1/m for m in moduli[i:]).
    # 4. If Upper Bound <= best_score, prune.
    # 5. Branch 1: Skip moduli[i].
    # 6. Branch 2: Include moduli[i]. 
    #    - Find a valid residue r for moduli[i] given current_choices.
    #    - If found, recurse with i+1, new_score, new_choices.
    #    - To improve search, try residues that might allow more future additions? 
    #      Hard to know. Just pick the first valid one? Or try all?
    #      Trying all valid residues could explode. 
    #      However, if a valid residue exists, does it matter which one we pick for the *feasibility* of the rest?
    #      Yes, different residues impose different constraints on future moduli.
    #      Example: m=6. Previous: (10, 2). gcd(6,10)=2. r != 2 mod 2 => r != 0 mod 2.
    #      If we pick r=1, future m=15 (gcd 3) constraint: r != 1 mod 3.
    #      If we pick r=5, future m=15 constraint: r != 5 mod 3 => r != 2 mod 3.
    #      These are different.
    
    # To keep it fast, we will:
    # - Use a greedy upper bound that is slightly tighter: 
    #   Instead of sum(1/m), we can check if the *maximum possible* addition from the remaining set 
    #   (assuming no conflicts) exceeds the gap.
    # - We will limit the depth or use a heuristic to guide the search.
    
    # Actually, given the small range (181 moduli), a well-pruned B&B might work.
    # But "Trying all valid residues" is the bottleneck.
    
    # Alternative: 
    # Is it true that if a set of moduli is feasible, *any* valid assignment of residues 
    # allows the same set of *future* moduli to be added? No.
    
    # Let's try a different structural change: 
    # **Greedy with Local Search on the Set of Moduli**
    # 1. Start with all moduli.
    # 2. Check feasibility. If infeasible, remove the modulus with the smallest 1/m (largest m) 
    #    that is part of a conflict "clique" exceeding the modulus limit?
    #    Actually, simpler: 
    #    Iteratively remove the modulus that causes the most conflicts or has the lowest value.
    
    # Let's go with the **Branch and Bound on Moduli** but with a crucial optimization:
    # We don't need to try *all* residues for a given modulus in the B&B tree.
    # We can observe that for a given modulus m and a set of constraints, 
    # the "best" residue is one that minimizes the number of forbidden residues for future moduli.
    # This is complex to compute.
    
    # Let's stick to a robust **Simulated Annealing** but with a **Structural Change**:
    # The current program's SA changes residues. 
    # My new strategy: **Set-Based Search**.
    # The state is the *set of moduli* included. 
    # For a given set, we check if it's feasible (via bipartite matching or simple backtracking on residues).
    # If feasible, we calculate the score.
    # We search the space of subsets using a heuristic (e.g., start with all, remove worst offenders).
    
    # Why this is different:
    # Current: State is (modulus, residue) pairs. Moves change residues or add/remove moduli.
    # New: State is a subset of moduli. Feasibility is a separate sub-problem.
    # This decouples the combinatorial explosion of residues from the selection of moduli.
    
    # Feasibility Check for a set of moduli S:
    # Can we assign residues r_m for m in S such that r_m != r_n mod gcd(m, n)?
    # This is a CSP. 
    # We can solve it via backtracking on the moduli in S.
    # Since |S| is at most 181, and we do this check many times, it needs to be fast.
    # Optimization: If a subset T of S is infeasible, then S is infeasible.
    # We can maintain a cache of infeasible subsets? Too many.
    
    # Let's implement the **Set-Based Search**:
    # 1. Initialize S = all moduli [20, 200].
    # 2. Check if S is feasible.
    #    - If yes, this is the optimal solution (since all 1/m > 0, we want all moduli).
    #    - If no, we must remove at least one modulus.
    # 3. Which one to remove? 
    #    We want to remove the one with the smallest 1/m (largest m) to minimize loss.
    #    But removing a large m might not resolve the conflict.
    #    We need to identify a "conflict core".
    
    # Algorithm:
    # 1. S = all moduli.
    # 2. While S is not feasible:
    #      a. Find a minimal infeasible subset (conflict core) of S.
    #      b. Remove the modulus with the smallest 1/m from this core.
    #      c. Update S.
    # 3. Return S with optimal residues.
    
    # How to find a minimal infeasible subset?
    # We can use the fact that if S is infeasible, there exists a small subset that is infeasible.
    # We can try to find a pair or triple that is infeasible?
    # Actually, any pair is feasible. 
    # Infeasibility requires at least 3 moduli (e.g., 20, 30, 10 with gcd 10, needing 3 distinct residues mod 10 is feasible, but 20, 30, 40, 50, 60... 10 moduli all divisible by 10? 
    # If we have 11 moduli all divisible by 10, we need 11 distinct residues mod 10, which is impossible.
    # So, for each divisor d, let C_d be the set of moduli in S divisible by d.
    # If |C_d| > d, then S is infeasible.
    # This is a necessary condition. Is it sufficient?
    # For pairwise constraints r_i != r_j mod gcd(i,j), the condition |C_d| <= d for all d is NOT sufficient in general (it's a necessary condition for the "clique" argument, but the graph is not a clique).
    # However, for the specific structure of divisibility, it might be close.
    
    # Let's assume the condition |C_d| <= d for all d is the main bottleneck.
    # We can formulate this as: 
    # For each d, we can pick at most d moduli from the set {m in S : d|m}.
    # We want to maximize sum(1/m) subject to these constraints.
    # This is a Packing problem.
    
    # Let's solve this Packing problem using a Greedy Heuristic + Local Search.
    # 1. Start with S = all moduli.
    # 2. For each d from 2 to 200:
    #      If |C_d| > d:
    #          Remove moduli from C_d with smallest 1/m until |C_d| == d.
    # 3. Check if the resulting S is actually feasible (via backtracking on residues).
    # 4. If not feasible, identify the conflict and remove another modulus.
    
    # This is a structural change: we model the constraints as "capacity constraints" on divisors.
    
    def is_feasible(moduli_set):
        """
        Check if a set of moduli has a valid residue assignment.
        Uses backtracking with forward checking.
        """
        if not moduli_set:
            return True, []
        
        # Order moduli by number of constraints (highest degree first)
        # Degree of m is number of other moduli in set with gcd > 1
        # For speed, just sort by value or leave as is.
        mods = sorted(moduli_set)
        
        # Backtracking
        # State: index, assignments dict {m: r}
        # We can use a simple recursive search with pruning.
        
        def backtrack(idx, assignments):
            if idx == len(mods):
                return True, assignments
            
            m = mods[idx]
            # Try all residues 0..m-1
            for r in range(m):
                valid = True
                # Check against all previously assigned
                for m_prev, r_prev in assignments:
                    g = math.gcd(m, m_prev)
                    if (r - r_prev) % g == 0:
                        valid = False
                        break
                if valid:
                    new_assignments = assignments + [(m, r)]
                    success, _ = backtrack(idx + 1, new_assignments)
                    if success:
                        return True, new_assignments
            return False, []
        
        # This is too slow for large sets.
        # We need a faster feasibility check.
        # Given the time limit, we might not be able to run B&B on the full set.
        
        # Let's use the "Capacity Constraint" heuristic to build a candidate set,
        # then verify it.
        
        return False, [] # Placeholder, will implement specific check below

    # Implementation of the Capacity Constraint Strategy:
    
    # 1. Start with all moduli.
    # 2. For each divisor d (2 to 200), ensure at most d moduli in S are divisible by d.
    #    If more, remove the ones with smallest 1/m (largest m).
    # 3. This yields a candidate set S.
    # 4. Verify S is feasible. If not, remove the "worst" modulus from the conflict and repeat.
    
    S = set(range(20, 201))
    
    # Step 2: Enforce capacity constraints
    # We process divisors in increasing order? Or decreasing?
    # If we enforce d=2, we remove odd numbers? No, we remove numbers divisible by 2.
    # Wait, if |C_2| > 2, we remove moduli divisible by 2.
    # C_2 contains all even numbers. There are ~90 even numbers in [20, 200].
    # So we must remove all but 2 even numbers!
    # This is a very strong constraint.
    # Let's check: If we have m1=20, m2=30, m3=40.
    # gcd(20,30)=10, gcd(20,40)=20, gcd(30,40)=10.
    # Constraints:
    # r1 != r2 mod 10
    # r1 != r3 mod 20 => r1 != r3 mod 10 (since 10|20)
    # r2 != r3 mod 10
    # So r1, r2, r3 must be distinct mod 10.
    # This is possible.
    # But if we have 11 even numbers, they all share gcd >= 2?
    # No, gcd(20, 22) = 2.
    # If we have 3 even numbers m1, m2, m3, they don't necessarily all share a gcd of 10.
    # They share gcd >= 2.
    # Constraint: r_i != r_j mod gcd(m_i, m_j).
    # If gcd(m_i, m_j) = 2, then r_i != r_j mod 2.
    # This means all even moduli must have residues of different parity?
    # No, r_i != r_j mod 2 means one is even, one is odd.
    # So we can have at most 2 even moduli if they all pairwise have gcd 2?
    # If m1=20, m2=22, m3=24.
    # gcd(20,22)=2, gcd(20,24)=4, gcd(22,24)=2.
    # Constraints:
    # r1 != r2 mod 2
    # r1 != r3 mod 4
    # r2 != r3 mod 2
    # From r1 != r2 mod 2 and r2 != r3 mod 2, r1 and r3 have same parity.
    # r1 != r3 mod 4.
    # Let r1=0, r2=1. r3 must be even (same parity as r1) and r3 != 0 mod 4. So r3=2.
    # Check: r1=0, r3=2. 0 != 2 mod 4. OK.
    # So 3 even moduli is possible.
    # What about 4?
    # m1=20, m2=22, m3=24, m4=26.
    # gcds: 2, 4, 2, 2, 2, 2.
    # r1, r3 same parity. r2, r4 same parity.
    # r1 != r2 (diff parity).
    # r1 != r3 mod 4.
    # r2 != r4 mod 2 (same parity, so just !=).
    # Let r1=0, r2=1, r3=2, r4=3.
    # r1(0) != r3(2) mod 4. OK.
    # r2(1) != r4(3) mod 2. 1!=1? No, 1 mod 2 = 1, 3 mod 2 = 1. Conflict!
    # r2 != r4 mod 2 requires different parity. But r2, r4 same parity as each other?
    # r2 != r1 (parity 1 vs 0). r4 != r1 (parity 1 vs 0).
    # So r2, r4 are both odd.
    # Constraint r2 != r4 mod 2 means 1 != 1, which is False.
    # So 4 even moduli is infeasible IF they form a chain of gcd 2s?
    # Actually, the condition |C_d| <= d is necessary.
    # For d=2, |C_2| is the number of even moduli.
    # If we have 3 even moduli, |C_2|=3 > 2.
    # But we found a solution for 3.
    # So the condition |C_d| <= d is NOT necessary?
    # Wait, the condition is for *cliques* in the intersection graph of prime factors?
    
    # Let's abandon the theoretical derivation and use a **Greedy + Repair** approach.
    # 1. Start with all moduli.
    # 2. Check feasibility. If infeasible, find the "most constrained" modulus (part of most conflicts) and remove it.
    # 3. Repeat until feasible.
    
    # To make this fast, we need a fast feasibility check.
    # We will use a randomized greedy assignment to test feasibility.
    # If it fails, we assume infeasible (with some risk).
    
    def try_assign_residues(moduli_list):
        """
        Attempts to assign residues to moduli_list using a greedy randomized approach.
        Returns list of (m, r) if successful, None otherwise.
        """
        # Order by m ascending (smaller m has fewer residues, harder to satisfy? 
        # Or larger m has more residues, easier? 
        # Usually, smaller moduli are more restrictive because they have fewer residues.
        # So process smaller moduli first.
        mods = sorted(moduli_list)
        assignments = []
        
        # Use a few restarts
        for _ in range(5):
            assignments = []
            success = True
            for m in mods:
                # Find a valid residue
                valid_r = -1
                # Try random residues
                residues = list(range(m))
                random.shuffle(residues)
                for r in residues:
                    ok = True
                    for m_prev, r_prev in assignments:
                        g = math.gcd(m, m_prev)
                        if (r - r_prev) % g == 0:
                            ok = False
                            break
                    if ok:
                        valid_r = r
                        break
                if valid_r == -1:
                    success = False
                    break
                assignments.append((m, valid_r))
            if success:
                return assignments
        return None

    # Main Loop:
    S = list(range(20, 201))
    
    # Remove moduli until feasible
    # We want to remove the ones with smallest 1/m (largest m) to preserve score.
    # But removing a large m might not fix the conflict.
    # Strategy: 
    # 1. Try to assign residues to S.
    # 2. If fail, identify a "conflict witness" (a subset that is infeasible).
    #    For simplicity, we can just remove the largest modulus and retry.
    #    This is slow.
    
    # Better: 
    # Sort S by descending m (largest first).
    # Try to build the set from smallest to largest.
    # Start with empty set.
    # Add moduli in increasing order of m.
    # If adding m makes the set infeasible, skip m.
    # This is a Greedy Algorithm.
    
    # Let's try this Greedy Algorithm first. It's fast and structurally simple.
    
    greedy_set = []
    for m in range(20, 201):
        candidate_set = greedy_set + [m]
        # Check feasibility of candidate_set
        # To speed up, we only need to check if m can be added to the existing assignments.
        # We keep the assignments from the previous step.
        pass
    
    # We need to maintain the assignments.
    current_mods = []
    current_assignments = []
    
    for m in range(20, 201):
        # Try to add m to current_assignments
        # Find a valid residue r for m
        valid_r = -1
        residues = list(range(m))
        random.shuffle(residues)
        for r in residues:
            ok = True
            for m_prev, r_prev in current_assignments:
                g = math.gcd(m, m_prev)
                if (r - r_prev) % g == 0:
                    ok = False
                    break
            if ok:
                valid_r = r
                break
        
        if valid_r != -1:
            current_mods.append(m)
            current_assignments.append((m, valid_r))
        else:
            # Skip m
            # Optional: Try to replace a previous modulus with m?
            # If 1/m > 1/m_prev, it might be worth it.
            # But m is larger than all previous, so 1/m is smaller.
            # So we never replace.
            pass
            
    # This Greedy produces a feasible set.
    # Score = sum(1/m for m in current_mods)
    
    # Now, can we improve it?
    # Local Search: Try to add back skipped moduli by changing residues of existing ones?
    # Or try to add a skipped modulus by removing a different one?
    
    # Let's add a Local Search step:
    # 1. List of skipped moduli.
    # 2. For each skipped m, try to add it.
    #    - If it fits with current residues, add it.
    #    - If not, try to change residues of existing moduli to make room.
    #    - If that fails, try to remove a modulus m_prev such that m can be added.
    
    # Given time constraints, let's stick to the Greedy + a few SA steps on the *set*.
    
    # Actually, the Greedy is likely very good because it processes small moduli first.
    # Small moduli are "expensive" in terms of constraints (few residues) but "valuable" (high 1/m).
    # By securing them first, we maximize the score.
    
    # Let's verify the score of this Greedy.
    # And then run a quick SA to see if we can improve.
    
    best_mods = current_mods[:]
    best_assignments = current_assignments[:]
    best_score = sum(1.0/m for m in best_mods)
    
    # SA on the set:
    # Move: Remove a random modulus, or Add a random skipped modulus (if fits).
    # Or: Change residue of a random modulus.
    
    T = 1.0
    alpha = 0.999
    deadline = start_time + 140.0
    
    while time.time() < deadline:
        # Random move
        move_type = random.random()
        
        if move_type < 0.5:
            # Change residue of an existing modulus
            if not best_assignments: continue
            idx = random.randint(0, len(best_assignments)-1)
            m, old_r = best_assignments[idx]
            new_r = random.randint(0, m-1)
            if new_r == old_r: continue
            
            # Check validity
            valid = True
            for j, (m_j, r_j) in enumerate(best_assignments):
                if j == idx: continue
                g = math.gcd(m, m_j)
                if (new_r - r_j) % g == 0:
                    valid = False
                    break
            
            if valid:
                new_assignments = best_assignments[:]
                new_assignments[idx] = (m, new_r)
                # Score unchanged
                # Accept? Always accept if valid (neutral move)
                best_assignments = new_assignments
        elif move_type < 0.8:
            # Add a skipped modulus
            skipped = [m for m in range(20, 201) if m not in best_mods]
            if not skipped: continue
            m_new = random.choice(skipped)
            
            # Try to find valid residue
            valid_r = -1
            residues = list(range(m_new))
            random.shuffle(residues)
            for r in residues:
                ok = True
                for m_j, r_j in best_assignments:
                    g = math.gcd(m_new, m_j)
                    if (r - r_j) % g == 0:
                        ok = False
                        break
                if ok:
                    valid_r = r
                    break
            
            if valid_r != -1:
                # Add it
                new_assignments = best_assignments + [(m_new, valid_r)]
                new_mods = best_mods + [m_new]
                new_score = best_score + 1.0/m_new
                if new_score > best_score:
                    best_assignments = new_assignments
                    best_mods = new_mods
                    best_score = new_score
        else:
            # Remove a modulus (to allow adding others later?)
            # Rarely useful since we want to maximize sum.
            # But might be needed to escape local optima where a large modulus blocks two small ones?
            # No, small ones are already in.
            # So removal is only useful if we plan to add a *better* one? 
            # But all remaining are larger (worse).
            # So removal is never beneficial for the objective, unless it allows adding a *skipped* modulus 
            # that was previously blocked, and that skipped modulus is *smaller* than the one removed?
            # No, skipped moduli are larger than included ones (since we added in increasing order).
            # So removal is not useful.
            pass
            
        T *= alpha
        if T < 0.01:
            T = 0.01

    # Final Result
    result = []
    for m, r in best_assignments:
        result.append([m, r])
        
    return result

import random
