"""Shortest superpermutation construction with greedy block reordering."""
import sys
import time
from itertools import permutations

# Deadline guard
_DEADLINE = 115.0  # seconds, safety margin before 120s limit
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def construct(n):
    """Construct a superpermutation on n symbols.
    
    Uses the classic recursive construction with a greedy Eulerian trail
    heuristic on the directed graph of permutation blocks, where edge
    selection prioritizes moves that keep the remaining unvisited edges
    in a single connected component (minimizing disconnected components).
    """
    global _start_time
    _start_time = time.time()
    
    if n <= 1:
        return "1"
    
    # Classic recursive construction
    word = [1]
    
    for m in range(2, n + 1):
        # Extract all distinct (m-1)-permutations from the current word
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Reorder perms using greedy Eulerian trail heuristic
        if len(perms) <= 500:
            perms = _greedy_eulerian_reorder(perms, m)
        
        # Build new word
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap with end of new
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        
        word = new
        
        # Check deadline
        if _elapsed() > _DEADLINE - 5:
            break
    
    return "".join(map(str, word))


def _greedy_eulerian_reorder(perms, m):
    """Reorder permutation blocks using a greedy Eulerian trail heuristic.
    
    Models the problem as finding a trail in a directed graph where:
    - Nodes are permutation blocks
    - Edges represent valid transitions with overlap costs
    
    The heuristic selects the next edge that:
    1. Maximizes overlap (minimizes added length)
    2. Among ties, minimizes the number of disconnected components in the
       subgraph of unvisited edges (keeping the graph connected)
    
    This is analogous to Hierholzer's algorithm but with a connectivity-aware
    greedy choice that avoids prematurely isolating unvisited edges.
    """
    if len(perms) <= 1:
        return perms
    
    n_perms = len(perms)
    
    # Precompute the full block for each perm
    blocks = [list(p) + [m] + list(p) for p in perms]
    
    # Compute overlap matrix: overlap[i][j] = max overlap of end of blocks[i] with start of blocks[j]
    overlap = [[0] * n_perms for _ in range(n_perms)]
    
    for i in range(n_perms):
        for j in range(n_perms):
            if i == j:
                continue
            bi = blocks[i]
            bj = blocks[j]
            max_ov = min(len(bi), len(bj))
            for o in range(max_ov, -1, -1):
                if o == 0 or bi[len(bi) - o:] == bj[:o]:
                    overlap[i][j] = o
                    break
    
    # Build adjacency structure: for each node, list of (neighbor, overlap_value)
    adj = [[] for _ in range(n_perms)]
    for i in range(n_perms):
        for j in range(n_perms):
            if i != j:
                adj[i].append((j, overlap[i][j]))
    
    # Greedy Eulerian trail with connectivity awareness
    best_order = None
    best_length = float('inf')
    
    # Try multiple starting points
    max_starts = n_perms if n_perms <= 30 else min(15, n_perms)
    
    for start in range(max_starts):
        if _elapsed() > _DEADLINE - 3:
            break
        
        order = [start]
        visited_edges = set()
        remaining = set(range(n_perms)) - {start}
        current_length = len(blocks[start])
        
        while remaining:
            last = order[-1]
            
            # Score each candidate next node
            best_next = None
            best_score = None
            
            for j in remaining:
                ov = overlap[last][j]
                
                # Compute connectivity score: how many disconnected components
                # would result in the subgraph of unvisited edges from j's perspective
                # Simplified: count how many remaining nodes are reachable from j
                # via unvisited edges (excluding the edge last->j we're about to use)
                
                # Quick connectivity check: count reachable nodes from j in remaining
                # using BFS/DFS on unvisited edges
                reachable = _count_reachable(j, remaining, adj, visited_edges, last, n_perms)
                
                # Score: primarily maximize overlap, secondarily maximize connectivity
                # We want high overlap AND high connectivity (fewer components)
                score = (ov, reachable)
                
                if best_score is None or score > best_score:
                    best_score = score
                    best_next = j
            
            if best_next is None:
                break
            
            order.append(best_next)
            remaining.remove(best_next)
            visited_edges.add((last, best_next))
            current_length += len(blocks[best_next]) - overlap[last][best_next]
        
        if len(order) == n_perms and current_length < best_length:
            best_length = current_length
            best_order = order
    
    if best_order is None:
        # Fallback: original order
        return perms
    
    # Apply 2-opt local search to improve the order
    if n_perms <= 100 and _elapsed() < _DEADLINE - 5:
        best_order, best_length = _two_opt_improve(best_order, overlap, blocks, n_perms)
    
    return [perms[i] for i in best_order]


def _count_reachable(start, remaining, adj, visited_edges, from_node, n_perms):
    """Count how many nodes in 'remaining' are reachable from 'start' via unvisited edges.
    
    This is a simplified connectivity metric that helps the greedy heuristic
    avoid choices that would fragment the remaining edge subgraph.
    """
    if not remaining:
        return 0
    
    # BFS from start, only traversing unvisited edges
    visited = set()
    queue = [start]
    visited.add(start)
    
    while queue:
        node = queue.pop(0)
        for neighbor, _ in adj[node]:
            if neighbor in remaining and neighbor not in visited:
                # Check if edge (node, neighbor) is unvisited
                if (node, neighbor) not in visited_edges:
                    visited.add(neighbor)
                    queue.append(neighbor)
    
    return len(visited & remaining)


def _two_opt_improve(order, overlap, blocks, n_perms):
    """Improve a permutation order using 2-opt local search.
    
    Tries reversing segments of the order to find better overlaps.
    Returns the improved order and its length.
    """
    current_order = list(order)
    current_length = _order_length(current_order, overlap, blocks)
    
    improved = True
    max_iterations = 50 if n_perms <= 50 else 20
    
    while improved and _elapsed() < _DEADLINE - 3:
        improved = False
        for _ in range(max_iterations):
            if _elapsed() > _DEADLINE - 3:
                break
            
            best_improvement = 0
            best_i, best_j = -1, -1
            
            # Try all 2-opt moves: reverse segment [i, j]
            for i in range(n_perms - 1):
                for j in range(i + 1, n_perms):
                    # Calculate improvement from reversing segment [i, j]
                    # Before: ... a[i-1] -> a[i] ... a[j] -> a[j+1] ...
                    # After:  ... a[i-1] -> a[j] ... a[i] -> a[j+1] ...
                    
                    old_len = 0
                    new_len = 0
                    
                    # Edge entering the segment
                    if i > 0:
                        old_len += len(blocks[current_order[i]]) - overlap[current_order[i-1]][current_order[i]]
                        new_len += len(blocks[current_order[j]]) - overlap[current_order[i-1]][current_order[j]]
                    else:
                        old_len += len(blocks[current_order[i]])
                        new_len += len(blocks[current_order[j]])
                    
                    # Edge leaving the segment
                    if j < n_perms - 1:
                        old_len += len(blocks[current_order[j+1]]) - overlap[current_order[j]][current_order[j+1]]
                        new_len += len(blocks[current_order[i]]) - overlap[current_order[i]][current_order[j+1]]
                    
                    # Internal edges (reversed)
                    for k in range(i + 1, j):
                        old_len += len(blocks[current_order[k]]) - overlap[current_order[k-1]][current_order[k]]
                        new_len += len(blocks[current_order[k]]) - overlap[current_order[k+1]][current_order[k]]
                    
                    improvement = old_len - new_len
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_i, best_j = i, j
            
            if best_improvement > 0:
                # Apply the best 2-opt move
                current_order[best_i:best_j+1] = reversed(current_order[best_i:best_j+1])
                current_length -= best_improvement
                improved = True
    
    return current_order, current_length


def _order_length(order, overlap, blocks):
    """Calculate the total length of concatenating blocks in the given order."""
    total = len(blocks[order[0]])
    for i in range(1, len(order)):
        total += len(blocks[order[i]]) - overlap[order[i-1]][order[i]]
    return total
