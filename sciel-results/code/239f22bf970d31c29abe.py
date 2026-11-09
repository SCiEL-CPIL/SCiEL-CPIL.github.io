"""
Greedy sequential vertex addition for a large graph with no regular induced
subgraph on 6+ vertices.

Strategy:
- Build the graph vertex-by-vertex. For each new vertex, choose its adjacency
  to existing vertices to minimise the number of new regular induced 6-subsets
  created.
- After construction, run a local repair that flips edges to eliminate any
  remaining regular induced subgraphs of size >= 6.
"""
import time
import random
import itertools
from collections import Counter

def construct():
    start = time.time()
    deadline = start + 135  # 15s safety margin before 150s kill

    best_n = 0
    best_edges = []

    # Try target sizes in descending order
    for target_n in [20, 19, 18, 17, 16, 15, 14, 13]:
        if time.time() > deadline - 15:
            break

        time_budget = min(40, deadline - 15 - (time.time() - start))
        if time_budget < 5:
            break

        rng = random.Random(42 + target_n)
        adj = _greedy_build(target_n, rng, deadline - 20)

        # Run local repair to fix any remaining violations
        if adj is not None:
            adj = _local_repair(adj, target_n, deadline - 10)
            if adj is not None and _is_valid(adj, target_n):
                if target_n > best_n:
                    best_n = target_n
                    best_edges = _adj_to_edges(adj, target_n)
                break

    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}

    # Fallback
    return {'n': 5, 'edges': []}


def _greedy_build(n, rng, deadline):
    """
    Greedily build a graph on n vertices. For each new vertex, evaluate all
    2^k possible adjacency patterns to existing vertices (k = current size)
    and pick the one that creates the fewest new regular 6-subsets.
    """
    adj = [0] * n
    for i in range(n):
        if time.time() > deadline:
            return None
        if i == 0:
            continue
        k = i  # number of existing vertices
        # If k is small, enumerate all 2^k patterns; otherwise random sample
        if k <= 10:
            best_mask = None
            best_count = float('inf')
            for mask in range(1 << k):
                # Count new regular 6-subsets involving vertex i
                count = _count_new_violations(adj, i, mask, k)
                if count < best_count:
                    best_count = count
                    best_mask = mask
                elif count == best_count and rng.random() < 0.3:
                    best_mask = mask
        else:
            # For larger k, try a few random masks
            best_mask = None
            best_count = float('inf')
            for _ in range(200):
                mask = rng.getrandbits(k)
                count = _count_new_violations(adj, i, mask, k)
                if count < best_count:
                    best_count = count
                    best_mask = mask
            if best_mask is None:
                best_mask = rng.getrandbits(k)

        # Apply the chosen mask
        for j in range(k):
            if (best_mask >> j) & 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj


def _count_new_violations(adj, new_v, mask, k):
    """
    Count the number of regular induced 6-subsets that include the new vertex
    'new_v' and 5 existing vertices, given the adjacency pattern 'mask'.
    """
    count = 0
    # The new vertex has degree within the 6-subset = number of bits set in
    # the mask restricted to the 5 chosen existing vertices + possibly 0
    # For each 5-subset of existing vertices:
    for subset5 in itertools.combinations(range(k), 5):
        # Degree of new_v in the induced 6-subset
        deg_new = 0
        for u in subset5:
            if (mask >> u) & 1:
                deg_new += 1
        # Degrees of the 5 existing vertices within the 6-subset
        degs = [deg_new]
        for u in subset5:
            d = 0
            for v in subset5:
                if u != v and (adj[u] & (1 << v)):
                    d += 1
            # Also add edge to new_v if present
            if (mask >> u) & 1:
                d += 1
            degs.append(d)
        # Check if all degrees are equal
        if all(d == degs[0] for d in degs):
            count += 1
    return count


def _local_repair(adj, n, deadline):
    """
    Local search: repeatedly find a regular induced subgraph of size >= 6
    and flip an edge in it to break regularity. Return None if we run out
    of time without finding a valid graph.
    """
    max_iters = 2000
    for it in range(max_iters):
        if time.time() > deadline:
            return None
        if _is_valid(adj, n):
            return adj
        # Find a violation
        viol = _find_violation(adj, n)
        if viol is None:
            return adj
        # Flip a random edge within the violation
        subset, size = viol
        if size >= 6:
            # Pick a random pair within the subset to flip
            pairs = list(itertools.combinations(range(size), 2))
            a, b = rng_choice(pairs)
            u = subset[a]
            v = subset[b]
            if adj[u] & (1 << v):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    return None


def rng_choice(seq):
    return seq[random.randint(0, len(seq) - 1)]


def _find_violation(adj, n):
    """Find any regular induced subgraph of size >= 6."""
    # Check size 6 first
    for subset in itertools.combinations(range(n), 6):
        if _is_regular(adj, subset):
            return (subset, 6)
    # Check larger sizes
    for size in range(7, n + 1):
        for subset in itertools.combinations(range(n), size):
            if _is_regular(adj, subset):
                return (subset, size)
    return None


def _is_regular(adj, subset):
    """Check if the induced subgraph on subset is regular."""
    size = len(subset)
    degs = []
    for v in subset:
        d = 0
        for u in subset:
            if u != v and (adj[v] & (1 << u)):
                d += 1
        degs.append(d)
    return all(d == degs[0] for d in degs)


def _is_valid(adj, n):
    """Check that no subset of size >= 6 induces a regular graph."""
    for size in range(6, n + 1):
        for subset in itertools.combinations(range(n), size):
            if _is_regular(adj, subset):
                return False
    return True


def _adj_to_edges(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return edges
