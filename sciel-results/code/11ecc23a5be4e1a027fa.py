"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _is_valid_set(S):
    """Check if set S satisfies the constraint."""
    S_list = sorted(S)
    for i, a in enumerate(S_list):
        for j in range(i+1, len(S_list)):
            b = S_list[j]
            for k in range(j, len(S_list)):
                c = S_list[k]
                if (b + c) % a == 0:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = 256
    random.seed(42)
    
    # Precompute all forbidden triples (a, b, c) with a < b <= c, a | (b+c)
    # Each triple is a constraint: NOT (a in S AND b in S AND c in S)
    # We want to maximize |S| subject to no forbidden triple being fully in S.
    
    # For each element i, precompute which other elements form violations with it
    # violation[i] = list of (j, k) such that (i,j,k) or (j,i,k) or (i,j,k) etc. is a forbidden triple
    
    # Actually, let's think of it differently.
    # A triple (a,b,c) with a<b<=c is forbidden if a|(b+c).
    # If all three are in S, it's a violation.
    # So we need: for every forbidden triple (a,b,c), at least one of a,b,c is NOT in S.
    
    # Let's precompute for each pair (b,c) with b<=c, which a's divide b+c
    # Then for each such a<b, we have a forbidden triple.
    
    # For the WalkSAT approach:
    # - State: boolean array x[1..n], x[i]=True means i is in S
    # - Objective: maximize sum(x[i])
    # - Constraints: for each forbidden triple (a,b,c), NOT(x[a] AND x[b] AND x[c])
    
    # Precompute forbidden triples as a list of (a,b,c) indices (0-based)
    # Also build for each variable, the list of clauses it appears in
    
    # A clause is violated when all its variables are True.
    # Clause: (a,b,c) is violated iff x[a]=x[b]=x[c]=True
    
    # Let's build the clause structure
    # clauses: list of tuples (a, b, c) - 0-based indices
    # var_clauses[i]: list of clause indices where variable i appears
    
    clauses = []
    var_clauses = [[] for _ in range(n)]
    
    for a in range(1, n+1):  # a is 1-based
        for b in range(a+1, n+1):
            for c in range(b, n+1):
                if (b + c) % a == 0:
                    clause_idx = len(clauses)
                    clauses.append((a-1, b-1, c-1))
                    var_clauses[a-1].append(clause_idx)
                    var_clauses[b-1].append(clause_idx)
                    var_clauses[c-1].append(clause_idx)
    
    num_clauses = len(clauses)
    
    # Initial solution: greedy from top
    x = [False] * n
    # Greedy: add from n down to 1 if no violation
    for cand in range(n-1, -1, -1):
        if time.time() > deadline:
            break
        valid = True
        for ci in var_clauses[cand]:
            a, b, c = clauses[ci]
            if x[a] and x[b] and x[c]:
                valid = False
                break
        if valid:
            x[cand] = True
    
    best_score = sum(x)
    best_x = x[:]
    
    # WalkSAT with dynamic clause weighting
    # Weights for each clause
    weights = [1.0] * num_clauses
    
    # For each variable, compute the number of violated clauses it's in
    def count_violations_for_var(var_idx, state):
        count = 0
        for ci in var_clauses[var_idx]:
            a, b, c = clauses[ci]
            if state[a] and state[b] and state[c]:
                count += 1
        return count
    
    def total_violations(state):
        count = 0
        for ci in range(num_clauses):
            a, b, c = clauses[ci]
            if state[a] and state[b] and state[c]:
                count += 1
        return count
    
    # Verify initial solution has no violations
    assert total_violations(x) == 0, f"Initial solution has {total_violations(x)} violations"
    
    # WalkSAT iterations
    # At each step:
    # 1. If no violations, try to add more elements (flip False->True if it doesn't create violations)
    # 2. If violations exist, pick a violated clause and flip a variable in it to try to satisfy it
    #    while minimizing loss in objective
    
    max_steps = 500000
    step = 0
    p_flip = 0.5  # probability of flipping a variable that reduces violations
    
    while time.time() < deadline and step < max_steps:
        step += 1
        
        if step % 10000 == 0 and time.time() > deadline - 5:
            break
        
        # Count current violations
        viol_count = total_violations(x)
        current_score = sum(x)
        
        if current_score > best_score:
            best_score = current_score
            best_x = x[:]
        
        if viol_count == 0:
            # No violations: try to add elements
            # Find all False variables that can be set to True without creating violations
            candidates = []
            for i in range(n):
                if not x[i]:
                    # Check if setting x[i]=True creates any violation
                    can_add = True
                    for ci in var_clauses[i]:
                        a, b, c = clauses[ci]
                        # After setting x[i]=True, is this clause violated?
                        # The clause is violated if all three are True
                        # x[i] will be True, so check if the other two are True
                        other_indices = [j for j in (a, b, c) if j != i]
                        if x[other_indices[0]] and x[other_indices[1]]:
                            can_add = False
                            break
                    if can_add:
                        candidates.append(i)
            
            if candidates:
                # Add the largest candidate (or random one)
                # Try adding multiple if possible
                random.shuffle(candidates)
                for i in candidates:
                    if time.time() > deadline:
                        break
                    # Re-check since other additions might have changed things
                    can_add = True
                    for ci in var_clauses[i]:
                        a, b, c = clauses[ci]
                        other_indices = [j for j in (a, b, c) if j != i]
                        if x[other_indices[0]] and x[other_indices[1]]:
                            can_add = False
                            break
                    if can_add:
                        x[i] = True
                        current_score += 1
                        if current_score > best_score:
                            best_score = current_score
                            best_x = x[:]
            else:
                # No single element can be added. Try removing one element to allow adding two.
                # For each element i in S, check: if we remove i, how many new elements can we add?
                # This is expensive, so do it occasionally
                if step % 1000 == 0:
                    found_improvement = False
                    for i in range(n-1, -1, -1):
                        if not x[i]:
                            continue
                        # Temporarily remove i
                        x[i] = False
                        # Find elements that can now be added
                        new_candidates = []
                        for j in range(n):
                            if not x[j]:
                                can_add = True
                                for ci in var_clauses[j]:
                                    a, b, c = clauses[ci]
                                    other_indices = [k for k in (a, b, c) if k != j]
                                    if x[other_indices[0]] and x[other_indices[1]]:
                                        can_add = False
                                        break
                                if can_add:
                                    new_candidates.append(j)
                        
                        if len(new_candidates) >= 2:
                            # Add them
                            for j in new_candidates:
                                x[j] = True
                            found_improvement = True
                            break
                        
                        # Restore i
                        x[i] = True
                    
                    if not found_improvement:
                        # Try a random flip to escape local optimum
                        i = random.randint(0, n-1)
                        x[i] = not x[i]
                        # Update weights for affected clauses
                        for ci in var_clauses[i]:
                            a, b, c = clauses[ci]
                            if x[a] and x[b] and x[c]:
                                weights[ci] = min(weights[ci] * 1.2, 100.0)
                            else:
                                weights[ci] = max(weights[ci] * 0.9, 0.1)
                        continue
        
        # Handle violations: pick a violated clause and flip a variable
        # Find all violated clauses
        violated = []
        for ci in range(num_clauses):
            a, b, c = clauses[ci]
            if x[a] and x[b] and x[c]:
                violated.append(ci)
        
        if not violated:
            continue
        
        # Pick a violated clause with highest weight
        max_w = max(weights[ci] for ci in violated)
        high_weight_violated = [ci for ci in violated if weights[ci] >= max_w * 0.8]
        ci = random.choice(high_weight_violated)
        a, b, c = clauses[ci]
        
        # For each variable in this clause, compute the impact of flipping it
        # Flipping a variable from True to False:
        #   - Satisfies this clause (removes 1 violation)
        #   - May create new violations in other clauses where this var was helping
        #   - Loses 1 from objective
        # Flipping from False to True: not applicable here since all are True in violated clause
        
        best_var = None
        best_score_delta = -1000000
        
        for var in (a, b, c):
            # Flip var from True to False
            # Objective change: -1
            # Violation change: this clause becomes satisfied (-1 violation)
            # But other clauses containing var might become violated
            new_violations = 0
            for ci2 in var_clauses[var]:
                if ci2 == ci:
                    continue
                a2, b2, c2 = clauses[ci2]
                # After flipping var to False, this clause is NOT violated (since var=False)
                # But we need to check if it WAS violated before and still is... 
                # Actually, if var is the only True variable that's being set to False,
                # the clause goes from potentially violated to not violated.
                # If var was True and the other two are True, it was violated, now it's not.
                # If var was True and one of the others is False, it wasn't violated, still isn't.
                # So flipping True->False can only REDUCE violations, never increase them.
                # Wait, that's not right for the other direction. Let me reconsider.
                
                # When we flip var from True to False:
                # - Any clause containing var that was violated (all three True) becomes satisfied
                # - No new violations are created (setting a var to False can't make a clause violated)
                # So the net effect on violations is: -1 for this clause, and possibly -1 for others
                # that were also violated.
                
                # Actually wait - I need to reconsider. The clause is violated when ALL three are True.
                # If I set one of them to False, the clause is no longer violated.
                # Setting a variable to False can only help satisfy clauses, never violate them.
                # So flipping True->False always reduces or maintains the violation count.
                
                pass
            
            # So flipping any True variable to False reduces violations by at least 1
            # and the objective by 1.
            # The question is: which flip allows us to add back more elements later?
            
            # For now, just pick the variable that's in the most violated clauses
            # (removing it satisfies the most clauses)
            num_violated_clauses = 0
            for ci2 in var_clauses[var]:
                a2, b2, c2 = clauses[ci2]
                if x[a2] and x[b2] and x[c2]:
                    num_violated_clauses += 1
            
            # Score: we want to maximize violations resolved, minimize objective loss
            # All flips lose 1 from objective, so maximize violations resolved
            if num_violated_clauses > best_score_delta:
                best_score_delta = num_violated_clauses
                best_var = var
        
        if best_var is not None:
            x[best_var] = False
            # Update weights
            for ci2 in var_clauses[best_var]:
                a2, b2, c2 = clauses[ci2]
                if x[a2] and x[b2] and x[c2]:
                    weights[ci2] = min(weights[ci2] * 1.1, 50.0)
                else:
                    weights[ci2] = max(weights[ci2] * 0.95, 0.1)
        
        # Occasionally try a different strategy: remove a low-utility element
        if step % 5000 == 0:
            # Find the element whose removal allows the most additions
            # This is expensive, so only do it occasionally
            best_remove = None
            best_net_gain = 0
            for i in range(n-1, -1, -1):
                if not x[i]:
                    continue
                x[i] = False
                # Count how many elements can be added
                addable = 0
                for j in range(n):
                    if not x[j]:
                        can_add = True
                        for ci2 in var_clauses[j]:
                            a2, b2, c2 = clauses[ci2]
                            other = [k for k in (a2, b2, c2) if k != j]
                            if x[other[0]] and x[other[1]]:
                                can_add = False
                                break
                        if can_add:
                            addable += 1
                x[i] = True
                net_gain = addable - 1  # lose i, gain addable
                if net_gain > best_net_gain:
                    best_net_gain = net_gain
                    best_remove = i
            
            if best_remove is not None and best_net_gain > 0:
                x[best_remove] = False
                # Add all possible elements
                for j in range(n):
                    if not x[j]:
                        can_add = True
                        for ci2 in var_clauses[j]:
                            a2, b2, c2 = clauses[ci2]
                            other = [k for k in (a2, b2, c2) if k != j]
                            if x[other[0]] and x[other[1]]:
                                can_add = False
                                break
                        if can_add:
                            x[j] = True
                current_score = sum(x)
                if current_score > best_score:
                    best_score = current_score
                    best_x = x[:]
    
    # Final check and cleanup
    # Ensure no violations in best_x
    viol = 0
    for ci in range(num_clauses):
        a, b, c = clauses[ci]
        if best_x[a] and best_x[b] and best_x[c]:
            viol += 1
    
    if viol > 0:
        # Fix violations by removing elements
        for ci in range(num_clauses):
            a, b, c = clauses[ci]
            if best_x[a] and best_x[b] and best_x[c]:
                # Remove the smallest element to preserve larger ones
                # Actually remove the one that's in the most other violated clauses
                counts = [0, 0, 0]
                for idx, var in enumerate((a, b, c)):
                    for ci2 in var_clauses[var]:
                        a2, b2, c2 = clauses[ci2]
                        if best_x[a2] and best_x[b2] and best_x[c2]:
                            counts[idx] += 1
                min_idx = min(range(3), key=lambda i: counts[i])
                best_x[(a, b, c)[min_idx]] = False
    
    # Final verification
    final_set = [i+1 for i in range(n) if best_x[i]]
    if not _is_valid_set(final_set):
        # Fallback: rebuild from scratch with simple greedy
        S = set()
        for cand in range(n, 0, -1):
            valid = True
            larger = [x for x in S if x > cand]
            for i in range(len(larger)):
                b = larger[i]
                for j in range(i, len(larger)):
                    c = larger[j]
                    if (b + c) % cand == 0:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                smaller = [x for x in S if x < cand]
                geq = [x for x in S if x >= cand]
                for a in smaller:
                    for c in geq:
                        if (cand + c) % a == 0:
                            valid = False
                            break
                    if not valid:
                        break
            if valid:
                for a in S:
                    if a >= cand:
                        continue
                    for b in S:
                        if a < b <= cand and (b + cand) % a == 0:
                            valid = False
                            break
                    if not valid:
                        break
            if valid:
                S.add(cand)
        final_set = sorted(S)
    
    return final_set
# EVOLVE-BLOCK-END
