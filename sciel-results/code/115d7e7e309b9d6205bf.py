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
        # Check distinct vertices
        if len(set(cycle)) != k:
            return False
        # Check all vertices in range
        for v in cycle:
            if v < 0 or v >= N:
                return False
        # Check consecutive edges exist
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
            # Quick check: verify each cycle
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
    # Structural idea from Inspiration 1: Base Hamiltonian cycle + chords from vertex 0
    # This creates a pancyclic graph with 59 edges (value 19).
    
    # 1. Base cycle edges: (0,1), (1,2), ..., (38,39), (39,0)
    edges = [(i, (i + 1) % N) for i in range(N)]
    
    # 2. Chords from 0 to k for k in 2..20
    # These chords allow forming cycles of length k+1 and N+1-k
    for k in range(2, 21):
        edges.append((0, k))
    
    # 3. Generate cycles of every length 3..40
    cycles = []
    for length in range(3, N + 1):
        if length == N:
            # Full Hamiltonian cycle: 0-1-...-39-0
            cycles.append(list(range(N)))
        else:
            k = length - 1
            if 2 <= k <= 20:
                # Cycle 0-1-2-...-k-0 (length k+1)
                # Uses edges (i, i+1) for i in 0..k-1 and (k, 0)
                cyc = list(range(k + 1))
            else:
                # Cycle 0-39-38-...-k-0 (length N+1-k)
                # Here k = length - 1 is not used directly in the same way.
                # We need a cycle of length L.
                # If L > 21, we use the "long" path from 0.
                # The chord used is (0, k') where k' = N + 1 - L.
                # The cycle is 0 - N-1 - ... - k' - 0
                # Length is (N - k') + 1 = N + 1 - k'.
                # We want length L, so k' = N + 1 - L.
                k_prime = N + 1 - length
                # Path: 0, N-1, N-2, ..., k_prime
                # Vertices: 0, 39, 38, ..., k_prime
                # Count: 1 (for 0) + (39 - k_prime + 1) = 40 - k_prime
                # Wait. 0 to 39 is 1 step. 39 to k_prime is 39-k_prime steps.
                # Total steps: 1 + (39 - k_prime) = 40 - k_prime.
                # We need length L.
                # 40 - k_prime = L => k_prime = 40 - L.
                # Let's re-verify the chord logic from Inspiration 1.
                # Inspiration 1: "Cycle 0-39-38-...-k-0 (length 41-k)"
                # If the cycle is 0, 39, ..., k, 0.
                # Vertices: 0, 39, 38, ..., k.
                # Number of vertices: 1 + (39 - k + 1) = 41 - k.
                # So length is 41 - k.
                # We want length L. So 41 - k = L => k = 41 - L.
                # The chord is (0, k).
                # So for L in 22..39, k = 41 - L is in 2..19.
                # This chord (0, k) was already added in the loop range(2, 21).
                k = 41 - length
                cyc = [0] + list(range(N - 1, k - 1, -1))
            cycles.append(cyc)
    
    # Optional: Prune edges (likely won't reduce count for this specific set, but safe)
    # edges = try_prune_edges(edges, cycles)
    
    return {'edges': edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
