import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    n = 40
    target_edges = 401
    
    def score_graph(adj):
        """Compute the largest book size."""
        max_book = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    # Count common neighbors
                    common = 0
                    for k in range(n):
                        if k != i and k != j and adj[i][k] and adj[j][k]:
                            common += 1
                    if common > max_book:
                        max_book = common
        return max_book
    
    def generate_initial():
        """Generate K_{20,20} + 1 internal edge."""
        adj = np.zeros((n, n), dtype=bool)
        for a in range(20):
            for b in range(20, 40):
                adj[a][b] = True
                adj[b][a] = True
        adj[0][1] = True
        adj[1][0] = True
        return adj
    
    adj = generate_initial()
    best_adj = adj.copy()
    best_score = score_graph(adj)
    
    # Local search: try swapping edges to reduce max book
    random.seed(42)
    
    def try_improve(adj, best_score):
        """Try random edge swaps to improve score."""
        adj = adj.copy()
        current_score = score_graph(adj)
        
        # Find all cross edges and internal edges
        cross_edges = []
        internal_edges_a = []
        internal_edges_b = []
        
        for i in range(20):
            for j in range(20, 40):
                if adj[i][j]:
                    cross_edges.append((i, j))
        
        for i in range(20):
            for j in range(i+1, 20):
                if adj[i][j]:
                    internal_edges_a.append((i, j))
        
        for i in range(20, 40):
            for j in range(i+1, 40):
                if adj[i][j]:
                    internal_edges_b.append((i, j))
        
        # Try swapping: remove a cross edge, add an internal edge (or vice versa)
        # or move internal edges around
        for _ in range(500):
            if time.time() > deadline:
                break
                
            # Strategy 1: Move an internal edge from a high-book position
            # Strategy 2: Swap a cross edge for an internal edge in a different part
            
            # Let's try: pick a random cross edge to remove, and a random non-edge internal to add
            if cross_edges and len(cross_edges) > 0:
                # Pick a random cross edge to potentially remove
                ci, cj = random.choice(cross_edges)
                
                # Find a random non-edge internal position to add
                # Try in part A
                if random.random() < 0.5:
                    # Part A internal
                    a1, a2 = random.sample(range(20), 2)
                    if not adj[a1][a2]:
                        # Remove cross edge (ci, cj), add internal (a1, a2)
                        adj[ci][cj] = False
                        adj[cj][ci] = False
                        adj[a1][a2] = True
                        adj[a2][a1] = True
                        
                        new_score = score_graph(adj)
                        if new_score < current_score:
                            current_score = new_score
                            if current_score < best_score:
                                best_score = current_score
                                best_adj = adj.copy()
                            # Update edge lists
                            cross_edges = [e for e in cross_edges if not (e[0]==ci and e[1]==cj)]
                            internal_edges_a.append((a1, a2))
                        else:
                            # Revert
                            adj[ci][cj] = True
                            adj[cj][ci] = True
                            adj[a1][a2] = False
                            adj[a2][a1] = False
                else:
                    # Part B internal
                    b1, b2 = random.sample(range(20, 40), 2)
                    if not adj[b1][b2]:
                        adj[ci][cj] = False
                        adj[cj][ci] = False
                        adj[b1][b2] = True
                        adj[b2][b1] = True
                        
                        new_score = score_graph(adj)
                        if new_score < current_score:
                            current_score = new_score
                            if current_score < best_score:
                                best_score = current_score
                                best_adj = adj.copy()
                            cross_edges = [e for e in cross_edges if not (e[0]==ci and e[1]==cj)]
                            internal_edges_b.append((b1, b2))
                        else:
                            adj[ci][cj] = True
                            adj[cj][ci] = True
                            adj[b1][b2] = False
                            adj[b2][b1] = False
            
            # Strategy: Move internal edge to different position
            if internal_edges_a:
                ii, ij = random.choice(internal_edges_a)
                # Try moving to a different pair in A
                if random.random() < 0.5:
                    a1, a2 = random.sample(range(20), 2)
                    if not adj[a1][a2] and (a1, a2) != (ii, ij) and (a2, a1) != (ii, ij):
                        old_score = current_score
                        adj[ii][ij] = False
                        adj[ij][ii] = False
                        adj[a1][a2] = True
                        adj[a2][a1] = True
                        
                        new_score = score_graph(adj)
                        if new_score < old_score:
                            current_score = new_score
                            if current_score < best_score:
                                best_score = current_score
                                best_adj = adj.copy()
                            internal_edges_a.remove((ii, ij))
                            internal_edges_a.append((a1, a2))
                        else:
                            adj[ii][ij] = True
                            adj[ij][ii] = False  # revert
                            adj[ij][ii] = True
                            adj[a1][a2] = False
                            adj[a2][a1] = False
            
            if internal_edges_b:
                bi, bj = random.choice(internal_edges_b)
                b1, b2 = random.sample(range(20, 40), 2)
                if not adj[b1][b2] and (b1, b2) != (bi, bj) and (b2, b1) != (bi, bj):
                    old_score = current_score
                    adj[bi][bj] = False
                    adj[bj][bi] = False
                    adj[b1][b2] = True
                    adj[b2][b1] = True
                    
                    new_score = score_graph(adj)
                    if new_score < old_score:
                        current_score = new_score
                        if current_score < best_score:
                            best_score = current_score
                            best_adj = adj.copy()
                        internal_edges_b.remove((bi, bj))
                        internal_edges_b.append((b1, b2))
                    else:
                        adj[bi][bj] = True
                        adj[bj][bi] = True
                        adj[b1][b2] = False
                        adj[b2][b1] = False
        
        return best_score, best_adj
    
    best_score, best_adj = try_improve(adj, best_score)
    
    # Convert to edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                edges.append([i, j])
    
    # Verify we have exactly 401 edges
    if len(edges) != target_edges:
        # Fallback to seed
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
    
    return edges
