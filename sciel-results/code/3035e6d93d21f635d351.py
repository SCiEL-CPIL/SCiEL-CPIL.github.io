import math
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    MAX_MOD = 128
    
    # The set of moduli is the set of all even numbers from 2 to 128.
    # Size = 64.
    # This is the largest possible clique in the non-coprime graph for this range.
    evens = [m for m in range(2, MAX_MOD + 1) if m % 2 == 0]
    n = len(evens)
    
    # Precompute gcds
    gcds = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            g = math.gcd(evens[i], evens[j])
            gcds[i][j] = g
            gcds[j][i] = g
    
    # Strategy: Randomized Local Search (Hill Climbing / Random Walk)
    # 1. Initialize with random residues.
    # 2. Find all conflicts.
    # 3. Pick a random conflict (i, j).
    # 4. Change r_i or r_j to a value that resolves this conflict (and hopefully doesn't create new ones).
    # 5. Repeat until no conflicts or time limit.
    
    def count_conflicts(residues):
        conflicts = []
        for i in range(n):
            for j in range(i + 1, n):
                g = gcds[i][j]
                if (residues[i] - residues[j]) % g == 0:
                    conflicts.append((i, j))
        return conflicts

    def is_valid(residues):
        return len(count_conflicts(residues)) == 0

    best_solution = None
    best_conflict_count = float('inf')
    
    # Try multiple random restarts
    num_restarts = 50
    time_per_restart = (deadline - start_time) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize randomly
        residues = [random.randint(0, evens[i] - 1) for i in range(n)]
        
        # Local search
        current_conflicts = count_conflicts(residues)
        current_count = len(current_conflicts)
        
        if current_count < best_conflict_count:
            best_conflict_count = current_count
            if current_count == 0:
                best_solution = residues[:]
                break
            else:
                best_solution = residues[:]
        
        # Hill Climbing with Random Restarts within the restart
        # We track visited states to avoid loops (simplified: just track last few)
        # Or just do random walks if stuck.
        
        stuck_count = 0
        local_deadline = time.time() + min(5.0, deadline - start_time)
        
        while time.time() < local_deadline:
            if current_count == 0:
                best_solution = residues[:]
                break
            
            if current_count < best_conflict_count:
                best_conflict_count = current_count
                best_solution = residues[:]
            
            if not current_conflicts:
                break
                
            # Pick a random conflict
            i, j = random.choice(current_conflicts)
            
            # Try to fix by changing i or j
            # Generate candidate values for i that don't conflict with j
            g = gcds[i][j]
            r_j = residues[j]
            
            # Values r_i such that (r_i - r_j) % g != 0
            valid_r_i = []
            for r in range(evens[i]):
                if (r - r_j) % g != 0:
                    valid_r_i.append(r)
            
            # Prefer values that might reduce other conflicts?
            # For now, just pick randomly from valid_r_i
            if valid_r_i:
                new_r_i = random.choice(valid_r_i)
                if new_r_i != residues[i]:
                    old_r_i = residues[i]
                    residues[i] = new_r_i
                    new_conflicts = count_conflicts(residues)
                    new_count = len(new_conflicts)
                    
                    if new_count <= current_count:
                        current_conflicts = new_conflicts
                        current_count = new_count
                        stuck_count = 0
                    else:
                        # Accept with some probability (Simulated Annealing style)
                        # or just revert
                        # Let's revert to keep it simple and deterministic-ish
                        residues[i] = old_r_i
                        stuck_count += 1
                        
            if stuck_count > 100:
                # Random kick
                k = random.randint(0, n - 1)
                residues[k] = random.randint(0, evens[k] - 1)
                current_conflicts = count_conflicts(residues)
                current_count = len(current_conflicts)
                stuck_count = 0
                
        if best_solution and is_valid(best_solution):
            break

    if best_solution and is_valid(best_solution):
        return [[best_solution[i], evens[i]] for i in range(n)]
    
    # Fallback: Greedy construction
    # This should rarely be needed if local search works.
    # But let's try a simple greedy to ensure we return *something* valid if search fails.
    # Greedy: Pick moduli in order, assign first valid residue.
    # This might not find 64, but will find a valid subset.
    
    selected_mods = []
    selected_res = []
    for m in evens:
        found = False
        for r in range(m):
            ok = True
            for idx, (m_p, r_p) in enumerate(zip(selected_mods, selected_res)):
                g = math.gcd(m, m_p)
                if (r - r_p) % g == 0:
                    ok = False
                    break
            if ok:
                selected_mods.append(m)
                selected_res.append(r)
                found = True
                break
        if not found:
            continue
            
    if selected_mods:
        return [[selected_res[i], selected_mods[i]] for i in range(len(selected_mods))]
        
    return [[1, 2]]

# EVOLVE-BLOCK-END
