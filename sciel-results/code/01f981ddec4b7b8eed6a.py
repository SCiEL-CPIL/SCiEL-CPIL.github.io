"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import sys
from itertools import combinations

def _paley_graph(q):
    """Return adjacency matrix (list of sets) for Paley graph on q vertices."""
    residues = {(x * x) % q for x in range(1, q)}
    adj = [set() for _ in range(q)]
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i].add(j)
                adj[j].add(i)
    return adj

def _compute_k4s(adj, n):
    """Return list of all K4s (tuples of 4 vertices) in the graph."""
    k4s = []
    for a in range(n):
        for b in range(a + 1, n):
            if b not in adj[a]:
                continue
            for c in range(b + 1, n):
                if c not in adj[a] or c not in adj[b]:
                    continue
                for d in range(c + 1, n):
                    if d in adj[a] and d in adj[b] and d in adj[c]:
                        k4s.append((a, b, c, d))
    return k4s

def _compute_independent_4sets(adj, n):
    """Return list of all independent 4-sets (tuples of 4 vertices) in the graph."""
    ind4s = []
    for a in range(n):
        for b in range(a + 1, n):
            if b in adj[a]:
                continue
            for c in range(b + 1, n):
                if c in adj[a] or c in adj[b]:
                    continue
                for d in range(c + 1, n):
                    if d not in adj[a] and d not in adj[b] and d not in adj[c]:
                        ind4s.append((a, b, c, d))
    return ind4s

def _find_safe_neighbor_set(adj, n, k4s, ind4s):
    """
    Find a set S of neighbors for a new vertex such that:
    - S contains no K4
    - V\\S contains no independent 4-set
    
    Uses iterative repair: start with a random S, then fix violations.
    Returns S as a set, or None if no valid S found after attempts.
    """
    vertices = set(range(n))
    
    for attempt in range(200):
        # Initialize S randomly
        if attempt == 0:
            # Try balanced start
            S = set(random.sample(range(n), n // 2))
        else:
            S = set()
            for i in range(n):
                if random.random() < 0.5:
                    S.add(i)
        
        # Iteratively fix violations
        for iteration in range(50):
            # Check if S contains any K4
            violated = False
            for k4 in k4s:
                if all(v in S for v in k4):
                    # Remove one random vertex from this K4 from S
                    v = random.choice(k4)
                    S.discard(v)
                    violated = True
                    break
            
            if violated:
                continue
            
            # Check if V\S contains any independent 4-set
            non_S = vertices - S
            violated = False
            for ind4 in ind4s:
                if all(v in non_S for v in ind4):
                    # Add one random vertex from this ind4 to S (removing from non_S)
                    v = random.choice(ind4)
                    S.add(v)
                    violated = True
                    break
            
            if not violated:
                # Check again to be sure
                ok = True
                for k4 in k4s:
                    if all(v in S for v in k4):
                        ok = False
                        break
                if ok:
                    for ind4 in ind4s:
                        if all(v not in S for v in ind4):
                            ok = False
                            break
                if ok:
                    return S
        # If we get here, this attempt failed
    
    return None

def _add_vertex(adj, n, S):
    """Add a new vertex n with neighbor set S. Returns new adjacency list."""
    new_adj = [set() for _ in range(n + 1)]
    for i in range(n):
        new_adj[i] = set(adj[i])
    new_adj[n] = set()
    for i in S:
        new_adj[i].add(n)
        new_adj[n].add(i)
    return new_adj

def _is_valid(adj, n):
    """Check if graph is (5,5)-free using K4 and independent-4set enumeration."""
    k4s = _compute_k4s(adj, n)
    # Check for K5: any vertex adjacent to a K4
    for v in range(n):
        for k4 in k4s:
            if all(x in adj[v] for x in k4):
                return False
    
    ind4s = _compute_independent_4sets(adj, n)
    # Check for independent 5: any vertex non-adjacent to an independent 4-set
    for v in range(n):
        for ind4 in ind4s:
            if all(x not in adj[v] for x in ind4):
                return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 310  # safety margin before 330s kill
    
    # Start with Paley graph on 37 vertices
    q = 37
    adj = _paley_graph(q)
    n = q
    
    # Verify initial graph is valid
    if not _is_valid(adj, n):
        return n, [(i, j) for i in range(n) for j in range(i + 1, n) if j in adj[i]]
    
    best_n = n
    best_adj = [set(s) for s in adj]
    
    # Precompute K4s and independent 4-sets
    k4s = _compute_k4s(adj, n)
    ind4s = _compute_independent_4sets(adj, n)
    
    # Try to grow the graph by adding vertices
    stall_count = 0
    while time.time() < deadline:
        if n >= 42:
            break
        
        # Find a safe neighbor set for the new vertex
        S = _find_safe_neighbor_set(adj, n, k4s, ind4s)
        if S is not None:
            adj = _add_vertex(adj, n, S)
            n += 1
            best_n = n
            best_adj = [set(s) for s in adj]
            # Recompute K4s and independent 4-sets
            k4s = _compute_k4s(adj, n)
            ind4s = _compute_independent_4sets(adj, n)
            stall_count = 0
        else:
            stall_count += 1
            # Try local search: flip edges to enable vertex addition
            found = False
            for _ in range(100):
                if time.time() >= deadline:
                    break
                
                # Flip a random edge
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                
                if j in adj[i]:
                    adj[i].discard(j)
                    adj[j].discard(i)
                else:
                    adj[i].add(j)
                    adj[j].add(i)
                
                # Recompute and try to add vertex
                k4s_new = _compute_k4s(adj, n)
                ind4s_new = _compute_independent_4sets(adj, n)
                S = _find_safe_neighbor_set(adj, n, k4s_new, ind4s_new)
                if S is not None:
                    adj = _add_vertex(adj, n, S)
                    n += 1
                    best_n = n
                    best_adj = [set(s) for s in adj]
                    k4s = _compute_k4s(adj, n)
                    ind4s = _compute_independent_4sets(adj, n)
                    found = True
                    break
                else:
                    # Revert the flip
                    if j in adj[i]:
                        adj[i].discard(j)
                        adj[j].discard(i)
                    else:
                        adj[i].add(j)
                        adj[j].add(i)
            
            if not found:
                # Reset to best if stuck for a while
                if stall_count > 5:
                    adj = [set(s) for s in best_adj]
                    n = best_n
                    k4s = _compute_k4s(adj, n)
                    ind4s = _compute_independent_4sets(adj, n)
                    stall_count = 0
    
    # Return the best graph found
    edges = [(i, j) for i in range(best_n) for j in range(i + 1, best_n) if j in best_adj[i]]
    return best_n, edges

# EVOLVE-BLOCK-END
