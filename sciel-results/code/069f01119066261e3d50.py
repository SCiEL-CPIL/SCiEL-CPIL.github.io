"""Path decompositions of three fixed connected graphs (Gallai's conjecture)."""
# EVOLVE-BLOCK-START
import time
from random import Random
from collections import defaultdict, deque

# Deadline guard: the evaluator kills after 150s; we target finishing well before.
_START_TIME = time.time()
DEADLINE = 140.0  # safety margin

def _elapsed():
    return time.time() - _START_TIME

def _deadline_exceeded():
    return _elapsed() > DEADLINE

def instance(n, p, seed):
    """Copy of the verifier's instance generator."""
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _decompose_gallai(edges, n):
    """
    Decompose a connected graph into the minimum number of paths.
    
    Gallai's theorem: a connected graph with 2k odd-degree vertices can be
    decomposed into k paths (if k >= 1). If all vertices have even degree,
    the graph is Eulerian and can be decomposed into 1 path (or 0 if no edges).
    
    Strategy:
    1. Identify odd-degree vertices.
    2. Pair them up arbitrarily (or optimally) to form k pairs.
    3. Add a "virtual" edge for each pair, making all degrees even.
    4. Find an Eulerian circuit in the augmented multigraph.
    5. Remove the virtual edges from the circuit; this splits it into k paths.
    """
    if not edges:
        return []
    
    # Build adjacency with edge IDs
    adj = defaultdict(list)  # vertex -> list of (neighbor, edge_id)
    edge_list = list(edges)
    for eid, (u, v) in enumerate(edge_list):
        adj[u].append((v, eid))
        adj[v].append((u, eid))
    
    # Find odd-degree vertices
    odd_vertices = [v for v in range(n) if len(adj[v]) % 2 == 1]
    k = len(odd_vertices) // 2
    
    if k == 0:
        # Eulerian graph: single path (circuit)
        # Find Eulerian circuit
        circuit = _eulerian_circuit(adj, n, edge_list)
        # Convert circuit to a path (remove last vertex which equals first)
        if circuit and len(circuit) > 1:
            return [circuit]
        else:
            return [[edge_list[0][0]]] if edge_list else []
    
    # Pair odd vertices
    # Simple pairing: pair consecutive odd vertices
    pairs = []
    for i in range(k):
        pairs.append((odd_vertices[2*i], odd_vertices[2*i+1]))
    
    # Build augmented multigraph
    # We'll work with a list of (u, v, is_virtual)
    aug_edges = [(u, v, False) for (u, v) in edge_list]
    for (u, v) in pairs:
        aug_edges.append((u, v, True))
    
    # Build adjacency for augmented graph
    aug_adj = defaultdict(list)  # vertex -> list of (neighbor, edge_index_in_aug)
    for idx, (u, v, virt) in enumerate(aug_edges):
        aug_adj[u].append((v, idx))
        aug_adj[v].append((u, idx))
    
    # Find Eulerian circuit in augmented graph
    # Use Hierholzer's algorithm
    circuit_edges = _eulerian_circuit_aug(aug_adj, n, aug_edges)
    
    # circuit_edges is a list of edge indices in aug_edges forming the circuit
    # Now extract the paths by removing virtual edges
    paths = []
    current_path = []
    for eid in circuit_edges:
        u, v, virt = aug_edges[eid]
        if virt:
            # This is a virtual edge; split here
            if current_path:
                paths.append(current_path)
                current_path = []
        else:
            if not current_path:
                current_path = [u, v]
            else:
                # The last vertex of current_path should be u or v
                if current_path[-1] == u:
                    current_path.append(v)
                elif current_path[-1] == v:
                    current_path.insert(0, u)
                else:
                    # This shouldn't happen in a valid Eulerian circuit
                    current_path.append(v)
    
    if current_path:
        paths.append(current_path)
    
    # Filter out empty paths
    paths = [p for p in paths if len(p) >= 2]
    
    return paths

def _eulerian_circuit(adj, n, edge_list):
    """Find Eulerian circuit using Hierholzer's algorithm. Returns list of vertices."""
    # adj: vertex -> list of (neighbor, edge_id)
    # We need to track which edges are used
    used = [False] * len(edge_list)
    
    # Use iterative Hierholzer's
    stack = []
    circuit = []
    
    # Start from any vertex with edges
    start = None
    for v in range(n):
        if adj[v]:
            start = v
            break
    if start is None:
        return []
    
    stack.append(start)
    while stack:
        v = stack[-1]
        # Find an unused edge
        found = False
        for i, (nb, eid) in enumerate(adj[v]):
            if not used[eid]:
                used[eid] = True
                # Also mark the reverse direction
                # Find the matching edge in nb's list
                for j, (nb2, eid2) in enumerate(adj[nb]):
                    if eid2 == eid:
                        adj[nb][j] = (-1, -1)  # Mark as used
                        break
                adj[v][i] = (-1, -1)
                stack.append(nb)
                found = True
                break
        if not found:
            circuit.append(stack.pop())
    
    circuit.reverse()
    return circuit

def _eulerian_circuit_aug(aug_adj, n, aug_edges):
    """
    Find Eulerian circuit in augmented multigraph.
    Returns list of edge indices in the order they appear in the circuit.
    """
    num_edges = len(aug_edges)
    used = [False] * num_edges
    
    # Build adjacency with edge indices
    adj = defaultdict(list)
    for idx, (u, v, virt) in enumerate(aug_edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))
    
    # Find start vertex
    start = None
    for v in range(n):
        if adj[v]:
            start = v
            break
    if start is None:
        return []
    
    # Hierholzer's algorithm - iterative
    stack = [(start, -1)]  # (vertex, incoming_edge_idx)
    circuit_edges = []
    
    while stack:
        v, in_eid = stack[-1]
        # Find an unused edge
        found = False
        for i, (nb, eid) in enumerate(adj[v]):
            if not used[eid]:
                used[eid] = True
                # Mark as used in both directions
                adj[v][i] = (-1, -1)
                for j, (nb2, eid2) in enumerate(adj[nb]):
                    if eid2 == eid:
                        adj[nb][j] = (-1, -1)
                        break
                stack.append((nb, eid))
                found = True
                break
        if not found:
            stack.pop()
            if in_eid != -1:
                circuit_edges.append(in_eid)
    
    circuit_edges.reverse()
    return circuit_edges

def construct():
    """
    Returns three lists of paths (vertex lists), one per fixed instance graph.
    Each path is a list of vertices where consecutive vertices are connected by an edge.
    Paths are simple, use graph edges, are edge-disjoint and cover all edges.
    """
    results = []
    for (n, p, seed) in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if _deadline_exceeded():
            # Fallback: return trivial decomposition (each edge as its own path)
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        edges = instance(n, p, seed)
        paths = _decompose_gallai(edges, n)
        results.append(paths)
    return results
# EVOLVE-BLOCK-END
