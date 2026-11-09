"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import sys

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    n = 256
    # Build conflict graph: edge (a,b) if (a*b) % (a+b) == 0
    # Vertices are 0..255 representing numbers 1..256
    adj = [set() for _ in range(n)]
    for i in range(n):
        a = i + 1
        for j in range(i + 1, n):
            b = j + 1
            if (a * b) % (a + b) == 0:
                adj[i].add(j)
                adj[j].add(i)
    
    # Precompute adjacency as bitsets
    adj_bits = [0] * n
    for i in range(n):
        for j in adj[i]:
            adj_bits[i] |= (1 << j)
    
    # Greedy coloring bound using bitsets (Tomita-style)
    def color_bound(remaining_bits):
        """Return upper bound on MIS size using greedy coloring with bitsets."""
        if remaining_bits == 0:
            return 0
        
        # Get list of vertices in remaining
        vlist = []
        temp = remaining_bits
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            vlist.append(v)
            temp ^= lsb
        
        # Sort by degree in remaining (descending) for better coloring
        vlist.sort(key=lambda v: -((adj_bits[v] & remaining_bits).bit_count()))
        
        # Greedy coloring
        color_of = [-1] * n
        max_color = 0
        for v in vlist:
            used = set()
            nb = adj_bits[v] & remaining_bits
            while nb:
                lsb = nb & (-nb)
                u = lsb.bit_length() - 1
                if color_of[u] >= 0:
                    used.add(color_of[u])
                nb ^= lsb
            c = 0
            while c in used:
                c += 1
            color_of[v] = c
            if c > max_color:
                max_color = c
        return max_color + 1
    
    # Greedy MIS for initial lower bound
    def greedy_mis():
        """Greedy: repeatedly pick vertex with fewest neighbors, include it, remove it and neighbors."""
        remaining = (1 << n) - 1
        result = 0
        while remaining:
            # Pick vertex with minimum degree in remaining subgraph
            best_v = -1
            best_deg = n + 1
            temp = remaining
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                deg = (adj_bits[v] & remaining).bit_count()
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
                temp ^= lsb
            result |= (1 << best_v)
            remaining &= ~(1 << best_v)
            remaining &= ~adj_bits[best_v]
        return result.bit_count(), result
    
    # Initial greedy solution
    best_size_global, best_sol_bits_global = greedy_mis()
    
    sys.setrecursionlimit(50000)
    
    # B&B search using bitsets with forward checking
    def bnb_search(remaining_bits, included_bits, size, depth=0):
        nonlocal best_size_global, best_sol_bits_global
        
        if time.time() - start_time > DEADLINE:
            return
        
        if remaining_bits == 0:
            if size > best_size_global:
                best_size_global = size
                best_sol_bits_global = included_bits
            return
        
        # Upper bound via greedy coloring
        ub = size + color_bound(remaining_bits)
        if ub <= best_size_global:
            return
        
        # Reduction: if any vertex in remaining has no neighbors in remaining,
        # it must be in the MIS
        changed = True
        while changed:
            changed = False
            rem = remaining_bits
            temp = rem
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                if (adj_bits[v] & rem) == 0:
                    # v must be included
                    included_bits |= lsb
                    size += 1
                    rem &= ~lsb
                    rem &= ~adj_bits[v]
                    changed = True
                temp ^= lsb
            remaining_bits = rem
        
        if remaining_bits == 0:
            if size > best_size_global:
                best_size_global = size
                best_sol_bits_global = included_bits
            return
        
        # Re-check upper bound after reduction
        ub = size + color_bound(remaining_bits)
        if ub <= best_size_global:
            return
        
        # Pick vertex with max degree in remaining for branching
        best_v = -1
        best_deg = -1
        temp = remaining_bits
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            deg = (adj_bits[v] & remaining_bits).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp ^= lsb
        
        v_bit = 1 << best_v
        
        # Branch 1: include v (usually more promising)
        new_remaining_1 = remaining_bits & ~v_bit & ~adj_bits[best_v]
        bnb_search(new_remaining_1, included_bits | v_bit, size + 1, depth + 1)
        
        # Check time before second branch
        if time.time() - start_time > DEADLINE:
            return
        
        # Branch 2: exclude v
        new_remaining_2 = remaining_bits & ~v_bit
        bnb_search(new_remaining_2, included_bits, size, depth + 1)
    
    # Start B&B
    bnb_search((1 << n) - 1, 0, 0)
    
    # Convert best solution back to set
    current = set()
    bits = best_sol_bits_global
    while bits:
        lsb = bits & (-bits)
        v = lsb.bit_length() - 1
        current.add(v)
        bits ^= lsb
    
    # Local search polish: try to add vertices that don't conflict
    improved = True
    while improved and (time.time() - start_time) < DEADLINE:
        improved = False
        for v in range(n):
            if v not in current:
                # Check if v conflicts with any vertex in current
                conflict = False
                for u in adj[v]:
                    if u in current:
                        conflict = True
                        break
                if not conflict:
                    current.add(v)
                    improved = True
    
    # Try swap improvements: remove one vertex, add all possible
    for _ in range(30):
        if (time.time() - start_time) > DEADLINE:
            break
        best_swap = None
        best_gain = 0
        for v in list(current):
            trial = set(current)
            trial.discard(v)
            added = 0
            for u in range(n):
                if u not in trial:
                    conflict = False
                    for w in adj[u]:
                        if w in trial:
                            conflict = True
                            break
                    if not conflict:
                        trial.add(u)
                        added += 1
            gain = added - 1
            if gain > best_gain:
                best_gain = gain
                best_swap = trial
        
        if best_swap:
            current = best_swap
        else:
            break
    
    # Final verification and cleanup
    final_set = set()
    for v in sorted(current):
        valid = True
        for u in final_set:
            if (adj_bits[v] >> u) & 1:
                valid = False
                break
        if valid:
            final_set.add(v)
    
    result = sorted(v + 1 for v in final_set)
    return result

# EVOLVE-BLOCK-END
