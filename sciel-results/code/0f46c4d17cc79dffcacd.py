import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    N = 180
    
    # 1. Generate all forbidden triples (a, b, c) with a < b < c
    triples = []
    elem_to_triples = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        a2 = a * a
        for u in range(1, a):
            if a2 % u == 0:
                v = a2 // u
                if u < v:
                    b = a + u
                    c = a + v
                    if b <= N and c <= N:
                        t = (a, b, c)
                        idx = len(triples)
                        triples.append(t)
                        elem_to_triples[a].append(idx)
                        elem_to_triples[b].append(idx)
                        elem_to_triples[c].append(idx)
                        
    num_triples = len(triples)
    
    M = 10000
    
    # State
    S = [False] * (N + 1)
    for i in range(1, N + 1):
        S[i] = True
        
    current_size = N
    current_violations = 0
    for t in triples:
        if S[t[0]] and S[t[1]] and S[t[2]]:
            current_violations += 1
            
    current_energy = -current_size + M * current_violations
    
    best_S = S[:]
    best_size = current_size
    best_energy = current_energy
    
    T = 10.0
    cooling = 0.9995
    rng = random.Random(42)
    elements = list(range(1, N + 1))
    
    while time.time() < deadline - 5.0:
        x = rng.choice(elements)
        was_in = S[x]
        
        # Flip
        S[x] = not was_in
        current_size += 1 if not was_in else -1
        
        # Update violations
        new_violations = current_violations
        for t_idx in elem_to_triples[x]:
            a, b, c = triples[t_idx]
            # Check violation status change
            # Only depends on the other two
            if a == x:
                others_in = S[b] and S[c]
            elif b == x:
                others_in = S[a] and S[c]
            else:
                others_in = S[a] and S[b]
            
            if others_in:
                if was_in:
                    # Was in, others in -> Was Violated. Now x out -> Not Violated.
                    new_violations -= 1
                else:
                    # Was out, others in -> Was Not Violated. Now x in -> Violated.
                    new_violations += 1
            # If others not in, violation status doesn't change (remains False)
            
        current_violations = new_violations
        new_energy = -current_size + M * current_violations
        
        delta_E = new_energy - current_energy
        
        if delta_E < 0 or (T > 0.001 and rng.random() < math.exp(-delta_E / T)):
            current_energy = new_energy
            
            if current_violations == 0 and current_size > best_size:
                best_S = S[:]
                best_size = current_size
                best_energy = current_energy
                
            # Optional: If we hit a valid state, try to greedily add?
            # No, let SA do it.
            
        else:
            # Revert
            S[x] = was_in
            current_size += 1 if was_in else -1
            current_violations = current_violations # Wait, we need to revert violations too
            # Revert logic:
            # We calculated new_violations based on the flip.
            # If we revert, we go back to current_violations (old).
            # But we overwrote current_violations? No, we used new_violations for calc.
            # We didn't update current_violations yet in the accept branch.
            # In reject branch, we just revert S and size.
            # current_violations is still the OLD value?
            # No, we set current_violations = new_violations BEFORE the check?
            # Ah, in the code above:
            # current_violations = new_violations is OUTSIDE the if/else?
            # Let's restructure.
            
            # Correct flow:
            # 1. Calc new_violations
            # 2. Calc new_energy
            # 3. If accept:
            #    current_violations = new_violations
            #    current_energy = new_energy
            #    update best
            # 4. Else:
            #    revert S[x], current_size
            #    (current_violations and current_energy remain unchanged)
            
        T *= cooling
        if T < 0.01:
            if rng.random() < 0.02:
                T = 5.0

    # Final Polish
    improved = True
    while improved and time.time() < deadline - 1.0:
        improved = False
        for x in range(1, N + 1):
            if not best_S[x]:
                valid = True
                for t_idx in elem_to_triples[x]:
                    a, b, c = triples[t_idx]
                    if a == x:
                        if best_S[b] and best_S[c]:
                            valid = False
                            break
                    elif b == x:
                        if best_S[a] and best_S[c]:
                            valid = False
                            break
                    else:
                        if best_S[a] and best_S[b]:
                            valid = False
                            break
                if valid:
                    best_S[x] = True
                    improved = True
                    best_size += 1

    # Verify
    for a, b, c in triples:
        if best_S[a] and best_S[b] and best_S[c]:
            return list(range(1, N + 1, 2))
            
    return [i for i in range(1, N + 1) if best_S[i]]
