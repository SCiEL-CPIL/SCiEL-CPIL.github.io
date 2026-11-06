import math
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # 1. Precompute all potential classes and their weights
    # Moduli range [20, 200]
    # For each modulus m, we need to pick a residue r.
    # However, the choice of r affects conflicts with other moduli.
    # This is not a simple graph on moduli, but a graph on (m, r) pairs.
    # Total nodes: sum_{m=20}^{200} m = 200*201/2 - 19*20/2 = 20100 - 190 = 19910 nodes.
    # This is too large for a naive MWIS on the full graph.
    
    # Alternative framing:
    # The constraint is r_i != r_j mod gcd(m_i, m_j).
    # This implies that for any pair of moduli m_i, m_j, the residues must be "compatible".
    # If we fix the set of moduli M, is there always a valid assignment of residues?
    # No. Example: m1=20, m2=30. gcd=10.
    # r1 in [0,19], r2 in [0,29].
    # Constraint: r1 % 10 != r2 % 10.
    # If we have m1=20, m2=30, m3=40.
    # gcd(20,40)=20. r1 % 20 != r3 % 20. Since r1 < 20, r1 != r3.
    # gcd(30,40)=10. r2 % 10 != r3 % 10.
    # 
    # Let's analyze the structure more closely.
    # If we choose moduli that are pairwise coprime, any residues work.
    # If moduli share factors, constraints arise.
    
    # A different approach:
    # The score is sum(1/m). To maximize this, we want small m.
    # The "bottleneck" is usually small moduli that conflict with many others.
    # 
    # Let's try a different structural insight:
    # If we select a set of moduli M, can we always assign residues?
    # Consider the graph where vertices are moduli.
    # Actually, let's look at the "residue class" representation.
    # 
    # Key Insight: 
    # If we pick moduli that are all even, say m_i = 2 * k_i.
    # gcd(m_i, m_j) is at least 2.
    # 
    # Let's try a simpler structural approach first:
    # **Lattice/Poset approach**:
    # If m1 | m2, then gcd(m1, m2) = m1.
    # Constraint: r1 % m1 != r2 % m1.
    # Since 0 <= r1 < m1, this means r1 != r2 % m1.
    # 
    # Let's try to formulate this as a Max-SAT or ILP? 
    # Too slow for 19910 variables.
    
    # Let's go back to the MWIS idea but with a twist.
    # The graph of (m, r) pairs is sparse.
    # Node (m, r) conflicts with (m', r') if r % gcd(m, m') == r' % gcd(m, m').
    # 
    # Can we reduce the state space?
    # For a fixed set of moduli M, the problem is to assign residues.
    # This is a CSP.
    # 
    # Let's try a **Genetic Algorithm** with a specific crossover and mutation that respects the structure, combined with a **Relaxation** to guide the search.
    # But the directive asks for a MATERIAL DIFFERENCE.
    
    # Let's try **Simulated Annealing** on the set of moduli, where the "energy" is the negative score, and the "feasibility" is checked by a fast CSP solver (backtracking) for the residues.
    # If a set of moduli is infeasible (no residue assignment), we penalize it heavily.
    # 
    # Actually, is any set of moduli feasible?
    # No. 
    # Counter-example:
    # m1=6, m2=10, m3=15. (Out of range, but illustrative)
    # gcd(6,10)=2. r1%2 != r2%2.
    # gcd(6,15)=3. r1%3 != r3%3.
    # gcd(10,15)=5. r2%5 != r3%5.
    # 
    # Let's check if there's a set of moduli in [20, 200] that is infeasible.
    # If m1, m2, m3 are such that their gcds create a contradiction.
    # 
    # Let's assume that for most "dense" sets of small moduli, a solution exists.
    # The current greedy approach finds a valid set.
    # 
    # **New Strategy: Branch and Bound on Moduli Selection**
    # We iterate through moduli in descending order of weight (1/m).
    # We maintain a set of selected moduli.
    # We use a **fast feasibility check** to see if the current set of moduli can be extended or if it's already infeasible.
    # 
    # Feasibility Check:
    # Given a set of moduli M, can we assign residues?
    # This is a constraint satisfaction problem.
    # We can use a simple backtracking solver for residues.
    # Since the number of moduli is likely < 50, and each has < 200 residues, this might be fast if we use constraint propagation.
    # 
    # However, doing this at every node of a B&B tree over 181 moduli is expensive.
    # 
    # Let's try a different angle: **Relaxation to Independent Set on Moduli Graph**
    # Define a graph G where vertices are moduli 20..200.
    # Edge (m1, m2) exists if it is "hard" to satisfy the constraint.
    # Actually, the constraint is always satisfiable for a pair.
    # The problem is global.
    
    # Let's look at the inspiration: "Simulated annealing and a final polish pass... Measured: 3.114 -> 3.149".
    # The current score is 2.9931.
    # The inspiration suggests SA works.
    # 
    # **Structural Change: Use a Max-Flow / Min-Cut formulation?**
    # No, this is not a flow problem.
    
    # **Structural Change: Use a Different Representation - "Residue Classes as Intervals"**
    # No.
    
    # **Structural Change: Use a "Conflict-Free" Basis**
    # Observe that if we choose moduli that are powers of 2, they are very restrictive.
    # If we choose moduli that are primes, they are very permissive (gcd=1).
    # 
    # Let's try a **Hybrid Approach**:
    # 1. Identify a core set of "hard" moduli (small ones) that must be handled carefully.
    # 2. Use a **CSP Solver (Backtracking with Forward Checking)** to find a valid residue assignment for a candidate set of moduli.
    # 3. Use **Simulated Annealing** to select the set of moduli.
    #    - State: Set of moduli.
    #    - Move: Add/Remove/Swap a modulus.
    #    - Cost: -Score + Penalty * (1 if Infeasible else 0).
    #    - Feasibility check: Fast CSP solver.
    
    # Why is this different?
    # The current program uses a greedy residue assignment which is myopic.
    # This program uses a global feasibility check (CSP) which allows it to evaluate the true feasibility of a modulus set, enabling it to explore sets that greedy might reject due to bad residue choices.
    
    def is_feasible(moduli):
        """
        Check if there exists a residue assignment for the given moduli.
        Uses backtracking with forward checking.
        """
        if not moduli:
            return True
        
        # Order moduli by number of constraints (most constrained first)
        # Constraint count for m_i is sum(1 for m_j in moduli if m_j != m_i)
        # Actually, just sort by size of gcds or something similar.
        # Let's just sort by modulus value ascending, as smaller moduli have fewer residues.
        sorted_moduli = sorted(moduli)
        
        # Assign residues in order
        assigned = {}
        
        def backtrack(idx):
            if idx == len(sorted_moduli):
                return True
            
            m = sorted_moduli[idx]
            # Try all residues for m
            for r in range(m):
                # Check consistency with already assigned moduli
                valid = True
                for j in range(idx):
                    m_j = sorted_moduli[j]
                    r_j = assigned[j]
                    g = math.gcd(m, m_j)
                    if r % g == r_j % g:
                        valid = False
                        break
                if valid:
                    assigned[idx] = r
                    if backtrack(idx + 1):
                        return True
                    # Backtrack
                    del assigned[idx]
            return False
        
        return backtrack(0)

    def get_residues(moduli):
        """
        Return a valid residue assignment for the given moduli.
        """
        if not moduli:
            return []
        
        sorted_moduli = sorted(moduli)
        assigned = {}
        
        def backtrack(idx):
            if idx == len(sorted_moduli):
                return True
            
            m = sorted_moduli[idx]
            for r in range(m):
                valid = True
                for j in range(idx):
                    m_j = sorted_moduli[j]
                    r_j = assigned[j]
                    g = math.gcd(m, m_j)
                    if r % g == r_j % g:
                        valid = False
                        break
                if valid:
                    assigned[idx] = r
                    if backtrack(idx + 1):
                        return True
                    del assigned[idx]
            return False
        
        if backtrack(0):
            # Map back to original order? No, just return list of [m, r]
            return [[sorted_moduli[i], assigned[i]] for i in range(len(sorted_moduli))]
        else:
            return None

    # Simulated Annealing
    all_moduli = list(range(20, 201))
    weights = {m: 1.0 / m for m in all_moduli}
    
    # Initial solution: Greedy (reuse logic from current program for initialization)
    def greedy_init():
        current = []
        remaining = all_moduli[:]
        random.shuffle(remaining)
        # Sort by weight desc for better start?
        remaining.sort(key=lambda m: weights[m], reverse=True)
        
        for m in remaining:
            # Try to add m with a valid residue
            # Simple check: try all residues
            found = False
            for r in range(m):
                ok = True
                for m_i, r_i in current:
                    g = math.gcd(m, m_i)
                    if r % g == r_i % g:
                        ok = False
                        break
                if ok:
                    current.append([m, r])
                    found = True
                    break
        return [m for m, _ in current]
    
    current_moduli = greedy_init()
    current_residues = get_residues(current_moduli)
    if current_residues is None:
        # Should not happen for greedy init
        current_moduli = []
        current_residues = []
    
    current_score = sum(weights[m] for m in current_moduli)
    best_moduli = current_moduli[:]
    best_residues = current_residues[:]
    best_score = current_score
    
    T = 1.0
    T_min = 0.001
    cooling = 0.999
    
    # Precompute gcds for speed?
    # gcds = [[math.gcd(m1, m2) for m2 in all_moduli] for m1 in all_moduli]
    # Index mapping: moduli index to value
    # val_to_idx = {m: i for i, m in enumerate(all_moduli)}
    
    steps = 0
    while time.time() < deadline - 5 and T > T_min:
        steps += 1
        if steps % 100 == 0 and time.time() > deadline - 2:
            break
            
        # Generate neighbor
        neighbor_moduli = current_moduli[:]
        if neighbor_moduli:
            if random.random() < 0.5:
                # Remove
                idx = random.randint(0, len(neighbor_moduli) - 1)
                neighbor_moduli.pop(idx)
            else:
                # Swap
                if len(neighbor_moduli) < len(all_moduli):
                    idx = random.randint(0, len(neighbor_moduli) - 1)
                    old_m = neighbor_moduli[idx]
                    new_m = random.choice([m for m in all_moduli if m not in neighbor_moduli])
                    neighbor_moduli[idx] = new_m
        else:
            # Add
            neighbor_moduli.append(random.choice(all_moduli))
        
        # Check feasibility
        neighbor_residues = get_residues(neighbor_moduli)
        
        if neighbor_residues is None:
            # Infeasible
            neighbor_score = -1000.0 # Large penalty
        else:
            neighbor_score = sum(weights[m] for m in neighbor_moduli)
        
        delta = neighbor_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / T):
            current_moduli = neighbor_moduli
            current_residues = neighbor_residues
            current_score = neighbor_score
            
            if current_score > best_score:
                best_score = current_score
                best_moduli = current_moduli[:]
                best_residues = current_residues[:]
        
        T *= cooling
        
    # Final Polish: Try to add any missing moduli
    if time.time() < deadline - 2:
        used = set(best_moduli)
        candidates = [m for m in all_moduli if m not in used]
        candidates.sort(key=lambda m: weights[m], reverse=True)
        
        for m in candidates:
            if time.time() > deadline - 2:
                break
            new_moduli = best_moduli + [m]
            new_residues = get_residues(new_moduli)
            if new_residues:
                new_score = best_score + weights[m]
                if new_score > best_score:
                    best_moduli = new_moduli
                    best_residues = new_residues
                    best_score = new_score
    
    # Return in required format
    if not best_residues:
        # Fallback
        best_residues = [[20, 0]]
        
    return best_residues

