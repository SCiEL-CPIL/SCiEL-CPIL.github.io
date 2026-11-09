import time
import random
import math

# Precomputation
D = 7
N = 2 ** D
TOTAL_EDGES = D * (2 ** (D - 1)) # 7 * 64 = 448

def _precompute():
    # Generate all edges
    edges = []
    edge_to_idx = {}
    for u in range(N):
        for i in range(D):
            v = u ^ (1 << i)
            if u < v:
                edge = (u, v)
                idx = len(edges)
                edges.append(edge)
                edge_to_idx[edge] = idx
    
    # Generate all faces (2-dimensional)
    # A face is defined by a vertex u and two distinct dimensions i, j
    # The 4 vertices are u, u^e_i, u^e_j, u^e_i^e_j
    # The 4 edges are (u, u^e_i), (u^e_i, u^e_i^e_j), (u^e_i^e_j, u^e_j), (u^e_j, u)
    # Note: u must be the "minimum" vertex in the face to avoid duplicates?
    # Actually, we can just iterate u and i<j, and ensure we count each face once.
    # A face is uniquely identified by the set of its 4 vertices.
    # Or simpler: iterate u, i, j. The face is determined by u, i, j.
    # But u, i, j and u^e_i, i, j define the same face.
    # To get unique faces, we can require that the bit i of u is 0? No, that doesn't work for all.
    # Standard way: A face is a subset of dimensions S with |S|=2 and a vertex u.
    # The face is the subcube induced by u and S.
    # Two pairs (u, S) and (v, T) define the same face if the subcubes are identical.
    # The subcube is determined by the fixed bits.
    # For a 2-face, 5 bits are fixed, 2 are free.
    # We can represent a face by the mask of free bits and the values of fixed bits.
    
    # Let's just generate them and use a set of frozensets of vertices to deduplicate.
    faces = []
    face_to_edges = {}
    edge_to_faces = {idx: [] for idx in range(TOTAL_EDGES)}
    
    face_set = set()
    
    for u in range(N):
        for i in range(D):
            for j in range(i + 1, D):
                v1 = u ^ (1 << i)
                v2 = u ^ (1 << j)
                v3 = u ^ (1 << i) ^ (1 << j)
                
                # Vertices: u, v1, v2, v3
                verts = frozenset([u, v1, v2, v3])
                if verts in face_set:
                    continue
                face_set.add(verts)
                
                # Edges of this face
                e1 = (min(u, v1), max(u, v1))
                e2 = (min(v1, v3), max(v1, v3))
                e3 = (min(v3, v2), max(v3, v2))
                e4 = (min(v2, u), max(v2, u))
                
                face_idx = len(faces)
                faces.append(verts)
                face_edges = [e1, e2, e3, e4]
                face_to_edges[face_idx] = face_edges
                
                for e in face_edges:
                    e_idx = edge_to_idx[e]
                    edge_to_faces[e_idx].append(face_idx)
                    
    return edges, edge_to_idx, faces, face_to_edges, edge_to_faces

# Global precomputed data
_EDGES, _EDGE_TO_IDX, _FACES, _FACE_TO_EDGES, _EDGE_TO_FACES = _precompute()
_NUM_FACES = len(_FACES)

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # State
    # active_edges: set of edge indices
    active = set()
    # face_counts: list of int, count of active edges in each face
    face_counts = [0] * _NUM_FACES
    
    def add_edge(idx):
        active.add(idx)
        for f in _EDGE_TO_FACES[idx]:
            face_counts[f] += 1
            
    def remove_edge(idx):
        active.remove(idx)
        for f in _EDGE_TO_FACES[idx]:
            face_counts[f] -= 1
            
    def can_add(idx):
        # Can add if no face containing this edge has count 3
        for f in _EDGE_TO_FACES[idx]:
            if face_counts[f] == 3:
                return False
        return True
        
    # 1. Greedy Initialization
    # Try to add all edges in random order? Or sorted?
    # Random order is better for diversification if we restart, but for single run,
    # a good greedy heuristic is to add edges that are in fewer "conflict" situations?
    # For now, simple random greedy.
    
    # We will do multiple restarts or just one long SA run?
    # Given 115s, one long SA run from a good start is likely best.
    # Let's do a greedy fill first.
    
    edge_indices = list(range(TOTAL_EDGES))
    random.shuffle(edge_indices)
    
    for idx in edge_indices:
        if can_add(idx):
            add_edge(idx)
            
    best_size = len(active)
    best_active = set(active)
    
    # 2. Simulated Annealing
    # We want to maximize len(active)
    # Current energy E = -len(active)
    # T starts at 1.0, cools to 0.01
    
    T = 1.0
    cooling_rate = 0.9999 # Tune this
    
    # Precompute list of inactive edges for faster sampling?
    # Or just pick random idx and check if it's in active.
    # TOTAL_EDGES is 448. Picking random idx is O(1).
    
    while time.time() < deadline:
        # Choose a move
        if random.random() < 0.5:
            # Try to add an edge
            idx = random.randint(0, TOTAL_EDGES - 1)
            if idx not in active and can_add(idx):
                add_edge(idx)
                # Always accept improvement
                if len(active) > best_size:
                    best_size = len(active)
                    best_active = set(active)
            # If not addable or already active, do nothing (or count as rejection)
        else:
            # Try to remove an edge
            if active:
                idx = random.choice(list(active))
                remove_edge(idx)
                # Delta E = +1 (worse)
                # Accept with prob exp(-1/T)
                if random.random() < math.exp(-1.0 / T):
                    # Keep removal
                    pass
                else:
                    # Revert removal
                    add_edge(idx)
        
        T *= cooling_rate
        if T < 0.01:
            T = 0.01
            
    # Return the best solution found
    out = []
    for idx in best_active:
        u, v = _EDGES[idx]
        out.append([u, v])
        
    return out
