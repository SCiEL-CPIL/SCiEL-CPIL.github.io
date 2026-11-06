"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

N = 40
START_TIME = time.time()
DEADLINE = 90  # seconds, keep margin

def elapsed():
    return time.time() - START_TIME

def check_validity(edges, cycles):
    """Check if all cycles are valid and all required edges exist."""
    edge_set = set()
    for e in edges:
        a, b = e
        if a == b:
            return False
        edge_set.add((min(a, b), max(a, b)))
    
    for k, cycle in enumerate(cycles, start=3):
        if len(cycle) != k:
            return False
        if len(set(cycle)) != k:
            return False
        for v in cycle:
            if v < 0 or v >= N:
                return False
        for i in range(k):
            u = cycle[i]
            v = cycle[(i + 1) % k]
            if (min(u, v), max(u, v)) not in edge_set:
                return False
    return True

def try_prune_edges(edges, cycles):
    """Try to remove edges one by one while maintaining validity."""
    edge_set = set()
    for e in edges:
        a, b = e
        edge_set.add((min(a, b), max(a, b)))
    
    current_edges = list(edge_set)
    changed = True
    while changed:
        changed = False
        for i in range(len(current_edges)):
            if elapsed() > DEADLINE - 5:
                break
            trial_edges = [e for j, e in enumerate(current_edges) if j != i]
            trial_set = set((min(a,b), max(a,b)) for a,b in trial_edges)
            valid = True
            for k, cycle in enumerate(cycles, start=3):
                for idx in range(k):
                    u = cycle[idx]
                    v = cycle[(idx + 1) % k]
                    if (min(u,v), max(u,v)) not in trial_set:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                current_edges = trial_edges
                changed = True
                break
        if elapsed() > DEADLINE - 5:
            break
    
    return [(a, b) for a, b in current_edges]

# EVOLVE-BLOCK-START
def construct():
    """
    Build a sparse pancyclic graph using a base cycle with strategic chords.
    
    Strategy:
    - Start with a Hamiltonian cycle (40 edges)
    - Add minimal chords to enable all cycle lengths 3..39
    - Use greedy chord selection to minimize total edge count
    """
    
    # Base: Hamiltonian cycle 0-1-2-...-39-0
    base_edges = set()
    for i in range(N):
        base_edges.add((i, (i+1) % N))
    
    # We need cycles of every length 3..40
    # The Hamiltonian cycle gives us length 40
    # For other lengths, we need chords that create shortcuts
    
    # Strategy: Add chords that are "shortcuts" on the cycle
    # A chord (i, j) where |i-j| mod N = d creates a cycle of length d+1
    # (going one way around the chord) and N-d+1 (going the other way)
    
    # We need all lengths 3..39, so we need chords that give us:
    # Length 3: chord (i, i+2) -> triangle i, i+1, i+2
    # Length 4: chord (i, i+3) -> 4-cycle
    # etc.
    
    # But we can be smarter: one chord can help create multiple cycle lengths
    # when combined with the base cycle structure
    
    # Let's use a greedy approach:
    # 1. Start with just the Hamiltonian cycle
    # 2. For each missing cycle length, find the cheapest chord to add
    # 3. Prune at the end
    
    # Actually, let's think about this differently.
    # A graph with a Hamiltonian cycle plus k chords has N+k edges.
    # We want to minimize k.
    
    # Key insight: If we have chords that partition the cycle into smaller cycles,
    # we can combine them to get intermediate lengths.
    
    # Let's try a structured approach:
    # Use chords that create a "wheel-like" structure or use power-of-2 chords
    
    # Alternative: Use a complete bipartite-like structure on the cycle
    # Chord (i, i+N/2) for all i creates many cycles
    
    # Let me try: add chords (i, i+2) for all i -> gives triangles
    # Then (i, i+4) for some i -> gives 4-cycles, etc.
    
    # Better approach: Use a recursive doubling structure
    # Chords: (i, i+2), (i, i+4), (i, i+8), (i, i+16) for appropriate i
    
    # Let's implement a greedy chord selection:
    
    edges = list(base_edges)
    edge_set = set(base_edges)
    
    # Track which cycle lengths we can form
    # A cycle of length L exists if there's a closed walk of length L using available edges
    # This is complex to check exactly, so let's use a heuristic:
    
    # For a cycle of length L, we need to find L distinct vertices v0,...,vL-1
    # such that (vi, vi+1 mod L) is an edge for all i
    
    # Greedy: for each missing length, try adding the "most useful" chord
    
    # Precompute: for each potential chord (i,j), which cycle lengths does it enable?
    # A chord (i,j) with distance d = min(|i-j|, N-|i-j|) enables:
    # - A cycle of length d+1 (using the chord and the short path)
    # - A cycle of length N-d+1 (using the chord and the long path)
    # - Potentially other lengths when combined with other chords
    
    # Let's be more systematic. We'll use local search:
    
    # Start with a good initial set of chords
    # Then try to remove chords that aren't needed
    
    # Initial chords: use a "binary" pattern
    # Chord (i, i+2^k) for k=1,2,3,4 and appropriate i
    
    chords = []
    
    # Add chords for small cycle lengths
    # Triangle: need (i, i+2)
    for i in range(0, N, 2):
        chords.append((i, (i+2) % N))
    
    # 4-cycle: need (i, i+3) or use two triangles
    for i in range(0, N, 4):
        chords.append((i, (i+4) % N))
    
    # 8-cycle
    for i in range(0, N, 8):
        chords.append((i, (i+8) % N))
    
    # 16-cycle
    for i in range(0, N, 16):
        chords.append((i, (i+16) % N))
    
    # Add all chords to edge set
    for c in chords:
        a, b = c
        edge_set.add((min(a,b), max(a,b)))
    
    edges = list(edge_set)
    
    # Now generate cycles for each length 3..40
    cycles = []
    
    for L in range(3, N+1):
        if L == N:
            # Hamiltonian cycle
            cycles.append(list(range(N)))
        else:
            # Find a cycle of length L
            # Strategy: use the base cycle and chords to create shortcuts
            # For length L, we want to "skip" N-L vertices
            
            # Simple approach: find L consecutive vertices on the cycle
            # and ensure they form a cycle (which they do if we have the right chords)
            
            # Actually, for a cycle of length L, we can use:
            # L-1 edges from the base cycle + 1 chord that closes the loop
            # Or use multiple chords
            
            # Let's try: take vertices 0, 1, ..., L-1
            # We need edge (L-1, 0) to close the cycle
            # This is a chord if L-1 != 1 and L-1 != N-1
            
            cycle = list(range(L))
            
            # Check if this cycle is valid with current edges
            valid = True
            for i in range(L):
                u = cycle[i]
                v = cycle[(i+1) % L]
                if (min(u,v), max(u,v)) not in edge_set:
                    valid = False
                    break
            
            if not valid:
                # Try different starting points or use chords
                # Find a valid cycle of length L
                found = False
                for start in range(N):
                    if elapsed() > DEADLINE - 10:
                        break
                    # Try cycle: start, start+1, ..., start+L-1 (mod N)
                    candidate = [(start + i) % N for i in range(L)]
                    
                    # Check if this is a valid cycle
                    cvalid = True
                    for i in range(L):
                        u = candidate[i]
                        v = candidate[(i+1) % L]
                        if (min(u,v), max(u,v)) not in edge_set:
                            cvalid = False
                            break
                    
                    if cvalid:
                        cycle = candidate
                        found = True
                        break
                
                if not found:
                    # Need to add a chord to make this work
                    # Find the missing edge and add it
                    for i in range(L):
                        u = cycle[i]
                        v = cycle[(i+1) % L]
                        if (min(u,v), max(u,v)) not in edge_set:
                            edge_set.add((min(u,v), max(u,v)))
                            edges = list(edge_set)
                            break
                    # Re-check
                    cvalid = True
                    for i in range(L):
                        u = cycle[i]
                        v = cycle[(i+1) % L]
                        if (min(u,v), max(u,v)) not in edge_set:
                            cvalid = False
                            break
                    if not cvalid:
                        # Fallback: use a different strategy
                        # Just use the first L vertices and hope for the best
                        pass
            
            cycles.append(cycle)
    
    # Prune edges
    edges = try_prune_edges(edges, cycles)
    
    return {'edges': edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
