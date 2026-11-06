"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

N = 20
# Precompute triples (a, b, c) with a < b < c
_TRIPLES = list(combinations(range(N), 3))
# For each triple, the mask of vertices NOT in the triple
_TRIPLE_MASKS = []
for (a, b, c) in _TRIPLES:
    mask = (1 << N) - 1
    mask &= ~(1 << a) & ~(1 << b) & ~(1 << c)
    _TRIPLE_MASKS.append(mask)

def _adj_to_masks(adj):
    """Convert n x n bool adjacency to list of int masks."""
    n = adj.shape[0]
    masks = [0] * n
    for i in range(n):
        m = 0
        for j in range(n):
            if adj[i, j]:
                m |= (1 << j)
        masks[i] = m
    return masks

def _masks_to_adj(masks):
    """Convert list of int masks to n x n bool adjacency."""
    n = len(masks)
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        m = masks[i]
        for j in range(n):
            if m & (1 << j):
                adj[i, j] = True
    return adj

def _count_violations(masks):
    """Count number of triples with >=3 common neighbors outside the triple."""
    count = 0
    for idx, (a, b, c) in enumerate(_TRIPLES):
        common = masks[a] & masks[b] & masks[c] & _TRIPLE_MASKS[idx]
        if bin(common).count('1') >= 3:
            count += 1
    return count

def _find_violation(masks):
    """Return (a,b,c,common_mask) for first violating triple, or None."""
    for idx, (a, b, c) in enumerate(_TRIPLES):
        common = masks[a] & masks[b] & masks[c] & _TRIPLE_MASKS[idx]
        if bin(common).count('1') >= 3:
            return (a, b, c, common)
    return None

def _is_k33_free(masks):
    """Check if graph is K3,3-free using fast bit operations."""
    for idx, (a, b, c) in enumerate(_TRIPLES):
        common = masks[a] & masks[b] & masks[c] & _TRIPLE_MASKS[idx]
        if bin(common).count('1') >= 3:
            return False
    return True

def _try_add(masks, u, v):
    """Try to add edge (u,v) to masks. Returns True if K3,3-free maintained."""
    if masks[u] & (1 << v):
        return True
    masks[u] |= (1 << v)
    masks[v] |= (1 << u)
    if not _is_k33_free(masks):
        masks[u] &= ~(1 << v)
        masks[v] &= ~(1 << u)
        return False
    return True

def _greedy_fill(masks, deadline):
    """Greedily add edges in random order until no more can be added or deadline."""
    n = N
    while True:
        if time.time() > deadline:
            break
        remaining = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (masks[i] & (1 << j)):
                    remaining.append((i, j))
        if not remaining:
            break
        random.shuffle(remaining)
        added = False
        for u, v in remaining:
            if time.time() > deadline:
                break
            if _try_add(masks, u, v):
                added = True
        if not added:
            break
    return masks

def _local_search(masks, deadline, max_iter=2000):
    """Local search: try adding edges; if stuck, try removing one edge and re-adding."""
    n = N
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        # Try to add any missing edge
        remaining = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (masks[i] & (1 << j)):
                    remaining.append((i, j))
        if not remaining:
            break
        random.shuffle(remaining)
        added = False
        for u, v in remaining:
            if time.time() > deadline:
                break
            if _try_add(masks, u, v):
                added = True
                break
        if not added:
            # Stuck: try removing a random existing edge to unlock
            existing = []
            for i in range(n):
                for j in range(i + 1, n):
                    if masks[i] & (1 << j):
                        existing.append((i, j))
            if not existing:
                break
            random.shuffle(existing)
            for u, v in existing[:8]:
                if time.time() > deadline:
                    break
                # Remove edge
                masks[u] &= ~(1 << v)
                masks[v] &= ~(1 << u)
                # Try to add more edges
                added2 = False
                remaining2 = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if not (masks[i] & (1 << j)):
                            remaining2.append((i, j))
                random.shuffle(remaining2)
                for u2, v2 in remaining2:
                    if time.time() > deadline:
                        break
                    if _try_add(masks, u2, v2):
                        added2 = True
                        break
                if not added2:
                    # Restore the removed edge
                    masks[u] |= (1 << v)
                    masks[v] |= (1 << u)
    return masks

def _edge_count(masks):
    """Count edges in graph represented by masks."""
    count = 0
    for i in range(N):
        count += bin(masks[i]).count('1')
    return count // 2

def _build_edges_from_masks(masks):
    """Convert masks to edge list."""
    edges = []
    for i in range(N):
        m = masks[i]
        for j in range(i + 1, N):
            if m & (1 << j):
                edges.append([i, j])
    return edges

def _prune_to_k33_free(masks, deadline):
    """Remove edges until K3,3-free, prioritizing removals that break violations."""
    while _find_violation(masks) is not None:
        if time.time() > deadline:
            break
        viol = _find_violation(masks)
        if viol is None:
            break
        a, b, c, common_mask = viol
        # Get list of common neighbor vertices
        common_verts = [v for v in range(N) if common_mask & (1 << v)]
        # Randomly remove an edge from the K3,3: pick one from {a,b,c} x common_verts
        side1 = random.choice([a, b, c])
        side2 = random.choice(common_verts)
        masks[side1] &= ~(1 << side2)
        masks[side2] &= ~(1 << side1)
    return masks

def construct():
    deadline = time.time() + 110  # safety margin before 120s limit

    best_masks = None
    best_count = 0

    # Start with seed: K2,18 (vertices 0,1 connected to 2..19)
    seed_masks = [0] * N
    for a in range(2):
        for b in range(2, N):
            seed_masks[a] |= (1 << b)
            seed_masks[b] |= (1 << a)
    best_count = _edge_count(seed_masks)
    best_masks = seed_masks[:]

    # Multi-restart: random initial graphs, prune, greedy fill, local search
    for restart in range(500):
        if time.time() > deadline:
            break

        # Random initial graph with varying density
        p = random.uniform(0.15, 0.6)
        masks = [0] * N
        for i in range(N):
            for j in range(i + 1, N):
                if random.random() < p:
                    masks[i] |= (1 << j)
                    masks[j] |= (1 << i)

        # Prune to K3,3-free
        masks = _prune_to_k33_free(masks, deadline)

        # Greedy fill
        masks = _greedy_fill(masks, deadline)

        # Local search
        masks = _local_search(masks, deadline, max_iter=300)

        ec = _edge_count(masks)
        if ec > best_count:
            best_count = ec
            best_masks = masks[:]

    # Final polish on best
    if best_masks is not None and time.time() < deadline:
        adj_best = _masks_to_adj(best_masks)
        masks_best = _adj_to_masks(adj_best)
        masks_best = _greedy_fill(masks_best, deadline)
        masks_best = _local_search(masks_best, deadline, max_iter=200)
        ec = _edge_count(masks_best)
        if ec > best_count:
            best_count = ec
            best_masks = masks_best

    return _build_edges_from_masks(best_masks)
# EVOLVE-BLOCK-END
