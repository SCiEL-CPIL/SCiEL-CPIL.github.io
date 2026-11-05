"""Simulated Annealing with incremental book-size updates to minimize the largest book."""
import time
import random
import numpy as np

_N = 40
_M = 401


def _init_adj():
    """K20,20 plus one intra-part edge, as adjacency matrix."""
    A = np.zeros((40, 40), dtype=np.int8)
    A[0:20, 20:40] = 1
    A[20:40, 0:20] = 1
    A[0, 1] = 1
    A[1, 0] = 1
    return A


def _edge_list(A):
    edges = []
    for u in range(40):
        for v in range(u + 1, 40):
            if A[u, v]:
                edges.append((u, v))
    return edges


def _compute_books(A):
    """Compute initial book sizes for all edges."""
    books = {}
    for u in range(40):
        for v in range(u + 1, 40):
            if A[u, v]:
                books[(u, v)] = int(np.sum(A[u] & A[v]))
    return books


def _max_book(books):
    if not books:
        return 0
    return max(books.values())


def construct():
    start = time.time()
    deadline = start + 115.0  # Safety margin

    rng = random.Random(42)

    A = _init_adj()
    books = _compute_books(A)
    edge_list = list(books.keys())
    
    # Build non_edges list
    edge_set = set(edge_list)
    non_edges = []
    for u in range(40):
        for v in range(u + 1, 40):
            if (u, v) not in edge_set:
                non_edges.append((u, v))

    cur_max = _max_book(books)
    best_max = cur_max
    best_A = A.copy()

    T = 5.0
    T_min = 0.01
    cooling = 0.9995

    while time.time() < deadline and T > T_min:
        if not edge_list or not non_edges:
            break
            
        u, v = rng.choice(edge_list)
        x, y = rng.choice(non_edges)
        
        # Identify affected edges for backup
        # Removal of (u,v): affects (u,w) and (v,w) for w in N(u)∩N(v)
        # Addition of (x,y): affects (x,z) and (y,z) for z in N(x)∩N(y)
        
        common_uv = np.where(A[u] & A[v] == 1)[0]
        affected = set()
        affected.add((u, v))
        for w in common_uv:
            if w == u or w == v: continue
            affected.add((min(u, w), max(u, w)))
            affected.add((min(v, w), max(v, w)))
            
        common_xy = np.where(A[x] & A[y] == 1)[0]
        affected.add((x, y))
        for z in common_xy:
            if z == x or z == y: continue
            affected.add((min(x, z), max(x, z)))
            affected.add((min(y, z), max(y, z)))

        # Backup values
        backup = {}
        for e in affected:
            if e in books:
                backup[e] = books[e]
            
        # Apply changes to books
        # 1. Remove (u,v)
        del books[(u, v)]
        
        for w in common_uv:
            if w == u or w == v: continue
            key_uw = (min(u, w), max(u, w))
            if key_uw in books:
                books[key_uw] -= 1
            key_vw = (min(v, w), max(v, w))
            if key_vw in books:
                books[key_vw] -= 1

        # 2. Add (x,y)
        books[(x, y)] = len(common_xy)
        
        for z in common_xy:
            if z == x or z == y: continue
            key_xz = (min(x, z), max(x, z))
            if key_xz in books:
                books[key_xz] += 1
            key_yz = (min(y, z), max(y, z))
            if key_yz in books:
                books[key_yz] += 1

        # Update A
        A[u, v] = 0
        A[v, u] = 0
        A[x, y] = 1
        A[y, x] = 1

        # Update lists
        edge_list.remove((u, v))
        edge_list.append((x, y))
        non_edges.remove((x, y))
        non_edges.append((u, v))

        # Calculate new max
        new_max = _max_book(books)
        
        # Acceptance
        delta = new_max - cur_max
        if delta <= 0 or rng.random() < np.exp(-delta / T):
            cur_max = new_max
            if cur_max < best_max:
                best_max = cur_max
                best_A = A.copy()
        else:
            # Revert
            A[u, v] = 1
            A[v, u] = 1
            A[x, y] = 0
            A[y, x] = 0
            
            edge_list.remove((x, y))
            edge_list.append((u, v))
            non_edges.remove((u, v))
            non_edges.append((x, y))
            
            # Restore books
            # Remove (x,y)
            if (x, y) in books:
                del books[(x, y)]
            # Restore (u,v)
            if (u, v) in backup:
                books[(u, v)] = backup[(u, v)]
                
            for e in affected:
                if e == (u, v) or e == (x, y):
                    continue
                if e in backup:
                    books[e] = backup[e]

        T *= cooling

    # Final: use best found
    Aout = best_A
    edges_out = _edge_list(Aout)
    if len(edges_out) != 401:
        Aout = _init_adj()
        edges_out = _edge_list(Aout)
    return [[u, v] for (u, v) in edges_out]
