import time
import math
import random
from fractions import Fraction

def construct():
    deadline = time.time() + 110  # Safety margin for 120s limit
    
    min_m = 20
    max_m = 200
    moduli = list(range(min_m, max_m + 1))
    
    # Precompute GCDs for speed
    gcd_table = {}
    for m1 in moduli:
        for m2 in moduli:
            gcd_table[(m1, m2)] = math.gcd(m1, m2)
            
    def check_valid(m, r, state):
        """Check if (m, r) is valid given state {m2: r2}"""
        for m2, r2 in state.items():
            if m2 == m: continue
            g = gcd_table.get((m, m2), math.gcd(m, m2))
            if r % g == r2 % g:
                return False
        return True
        
    def greedy_construct(randomize=False, order=None):
        """
        Greedy construction.
        If order is None, uses ascending moduli.
        """
        if order is None:
            order = moduli
            
        state = {}
        for m in order:
            if time.time() > deadline - 10:
                break
                
            # Find valid residues
            valid_rs = []
            for r in range(m):
                if check_valid(m, r, state):
                    valid_rs.append(r)
                    
            if not valid_rs:
                continue
                
            if randomize:
                r = random.choice(valid_rs)
            else:
                # Heuristic: pick r that minimizes "damage" to future moduli
                # Damage is roughly sum of (1/m_future) * (number of future residues blocked)
                # Number of residues of m2 blocked by (m, r) is m2 // gcd(m, m2)
                # But we only care about m2 > m typically, or all remaining.
                # Simpler: minimize sum of 1/m2 for m2 in remaining moduli where gcd is small?
                # Actually, if gcd is large, fewer residues are blocked? No, m2/g is the count.
                # If gcd is 1, m2 residues are blocked (all of them? No, 1 residue class mod 1 is 0, so all m2 residues are in class 0 mod 1. So yes, if gcd=1, ALL residues of m2 are blocked? 
                # Wait. r % 1 = 0. r2 % 1 = 0. So 0==0. Conflict!
                # So if gcd(m, m2) == 1, ANY r and r2 will conflict?
                # Let's re-read the condition: r_1 != r_2 modulo gcd(m_1, m_2).
                # If gcd=1, modulo 1, everything is 0. So 0 != 0 is False.
                # So ANY pair of coprime moduli conflicts?
                # "pairwise disjoint (r_1 != r_2 modulo gcd(m_1, m_2))"
                # If gcd=1, the only residue class mod 1 is {0, 1, ...} all equivalent to 0.
                # So r1 % 1 = 0 and r2 % 1 = 0. 0 == 0.
                # So yes, coprime moduli CANNOT be in the same set?
                # Let's check the knowledge: "Switching from a restricted coprime-modulus pool ... to a global greedy search ... allows packing multiple non-coprime moduli (e.g., evens) with disjoint residues".
                # This implies coprime moduli DO conflict.
                # So we should prioritize moduli with high gcds with others?
                # Or rather, we should pick moduli that are NOT coprime with the selected ones.
                # But the greedy loop iterates through ALL moduli. If it hits a coprime one, it will fail to find a valid residue and skip it.
                # This is correct behavior.
                
                # Heuristic for r selection:
                # We want to leave as many options open for other moduli as possible.
                # The number of residues of m2 blocked by (m, r) is m2 // gcd(m, m2).
                # This count is independent of r!
                # So any valid r blocks the same NUMBER of residues for any given m2.
                # However, the SPECIFIC residues blocked might matter if those residues are "good" for m2?
                # Since we just need ANY valid residue for m2, the count is the only thing that matters for feasibility.
                # So random choice is fine, or first choice.
                r = valid_rs[0]
                
            state[m] = r
            
        return state

    def local_search(state, max_iter=200):
        """
        Local search to improve the set of moduli.
        Tries to add missing moduli by changing residues of existing ones or swapping.
        """
        current_state = dict(state)
        current_score = sum(Fraction(1, m) for m in current_state)
        
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            improved = False
            
            # 1. Try to add any unselected modulus
            # Sort unselected by 1/m descending (smallest m first)
            unselected = [m for m in moduli if m not in current_state]
            unselected.sort() # Smallest m first (highest weight)
            
            added = False
            for m in unselected:
                # Can we add m with any residue?
                valid_rs = []
                for r in range(m):
                    if check_valid(m, r, current_state):
                        valid_rs.append(r)
                        
                if valid_rs:
                    current_state[m] = valid_rs[0]
                    current_score += Fraction(1, m)
                    improved = True
                    added = True
                    break # Restart loop as state changed
                    
            if added:
                continue
                
            # 2. Try to swap a selected modulus with a smaller unselected one
            # Or just try to remove a large m and see if a smaller m fits?
            # Actually, simply trying to add is often sufficient if we start with a good greedy.
            # But we might be stuck with a set of large moduli where a smaller one fits if we remove a large one.
            
            # Try swapping: remove m_old, try to add m_new < m_old
            # This is expensive. Let's try a simpler move:
            # Change residue of an existing modulus to see if it unlocks a new modulus?
            # This is complex.
            
            # Alternative: Random restart with different order?
            # Or just return if no improvement.
            
            if not improved:
                # Try a "kick": randomly change a residue of an existing modulus
                # and see if it allows adding a new one?
                if current_state:
                    m_var = random.choice(list(current_state.keys()))
                    old_r = current_state[m_var]
                    # Try a random new residue
                    new_r = random.randint(0, m_var - 1)
                    if new_r != old_r:
                        # Check if new_r is valid against others
                        temp_state = dict(current_state)
                        temp_state[m_var] = new_r
                        if check_valid(m_var, new_r, {k:v for k,v in temp_state.items() if k != m_var}):
                            current_state = temp_state
                            improved = True # Force re-evaluation of additions
                            
            if not improved:
                break
                
        return current_state, float(current_score)

    best_solution = []
    best_score = 0.0
    
    # Strategy:
    # 1. Run deterministic greedy (ascending moduli).
    # 2. Run local search on it.
    # 3. Run several randomized greedy restarts (shuffle moduli order).
    # 4. Run local search on those.
    
    # Run 1: Deterministic
    state1 = greedy_construct(randomize=False)
    sol1, score1 = local_search(state1)
    if score1 > best_score:
        best_score = score1
        best_solution = sol1
        
    # Run 2-10: Randomized restarts
    # Use a fixed number of restarts to ensure we finish in time.
    num_restarts = 20
    
    for i in range(num_restarts):
        if time.time() > deadline - 15:
            break
            
        # Random order of moduli
        order = moduli[:]
        random.shuffle(order)
        
        state = greedy_construct(randomize=True, order=order)
        sol, score = local_search(state)
        
        if score > best_score:
            best_score = score
            best_solution = sol
            
    # Convert to list of lists
    if best_solution:
        return [[m, r] for m, r in best_solution.items()]
    
    # Fallback
    state = greedy_construct(randomize=False)
    return [[m, r] for m, r in state.items()]

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    score = sum(1.0 / m for m, _ in res)
    print(f"Score: {score}")
    print(f"Time: {end - start:.2f}s")
    print(f"Classes: {len(res)}")
