import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    def count_k222_with_edge(adj, n, u, v, color):
        """
        Count K2,2,2 subgraphs in the graph defined by adj (color 'color')
        that contain the edge (u, v).
        adj[i][j] is 1 if edge (i,j) is color 'color', 0 otherwise.
        """
        if adj[u][v] != 1:
            return 0
        
        # S_u: neighbors of u in color 'color'
        # S_v: neighbors of v in color 'color'
        # We need a' in S_v, b' in S_u, a'!=u, b'!=v, a'!=b'
        # (a', b') must be an edge in color 'color'
        # c1, c2 in S_u intersect S_v, distinct from u,v,a',b'
        
        count = 0
        # Iterate a'
        # a' must be in S_v, so adj[v][a'] == 1
        # a' != u, a' != v
        for a_prime in range(n):
            if a_prime == u or a_prime == v:
                continue
            if adj[v][a_prime] != 1:
                continue
            
            # Iterate b'
            # b' must be in S_u, so adj[u][b'] == 1
            # b' != u, b' != v, b' != a_prime
            for b_prime in range(n):
                if b_prime == u or b_prime == v or b_prime == a_prime:
                    continue
                if adj[u][b_prime] != 1:
                    continue
                
                # Check edge (a_prime, b_prime)
                if adj[a_prime][b_prime] != 1:
                    continue
                
                # Find candidates for c1, c2
                # Must be in S_u and S_v
                # And distinct from u, v, a_prime, b_prime
                candidates = 0
                for c in range(n):
                    if c == u or c == v or c == a_prime or c == b_prime:
                        continue
                    if adj[u][c] == 1 and adj[v][c] == 1:
                        candidates += 1
                
                if candidates >= 2:
                    # Number of ways to choose 2 from candidates
                    count += candidates * (candidates - 1) // 2
        
        return count

    def total_k222_count(adj, n, color):
        """
        Count all K2,2,2 in color 'color'.
        To avoid double counting, we can sum over edges and divide by something?
        Each K2,2,2 has 12 edges.
        If we sum count_k222_with_edge over all edges, each K2,2,2 is counted 12 times.
        So total = sum / 12.
        """
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j] == 1:
                    total += count_k222_with_edge(adj, n, i, j, color)
        return total // 12

    def solve_for_n(n):
        # Initialize random coloring
        # adj[0] and adj[1] are n x n matrices
        adj = [[0]*n for _ in range(2)]
        
        # Random init
        for i in range(n):
            for j in range(i + 1, n):
                c = random.randint(0, 1)
                adj[c][i][j] = 1
                adj[c][j][i] = 1
                adj[1-c][i][j] = 0
                adj[1-c][j][i] = 0
        
        # Calculate initial counts
        count0 = total_k222_count(adj, n, 0)
        count1 = total_k222_count(adj, n, 1)
        best_count = count0 + count1
        best_adj = [row[:] for row in adj[0]] # copy? No, need to store both
        best_adj = [[row[:] for row in adj[0]], [row[:] for row in adj[1]]]
        
        if best_count == 0:
            return best_adj
        
        # Local Search
        temp = 1.0
        cooling = 0.999
        
        while time.time() < deadline:
            # Pick random edge
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            
            # Ensure i < j for indexing consistency in our storage?
            # Our storage is symmetric, so it doesn't matter for adjacency check.
            # But for flipping, we need to know current color.
            if adj[0][i][j] == 1:
                c_old = 0
                c_new = 1
            else:
                c_old = 1
                c_new = 0
            
            # Calculate gain/loss
            # Loss: K2,2,2 in c_old that use (i,j)
            loss = count_k222_with_edge(adj, n, i, j, c_old)
            
            # To calculate gain, we need the graph AFTER flip.
            # Temporarily flip
            adj[c_old][i][j] = 0
            adj[c_old][j][i] = 0
            adj[c_new][i][j] = 1
            adj[c_new][j][i] = 1
            
            gain = count_k222_with_edge(adj, n, i, j, c_new)
            
            # Update counts
            new_count0 = count0 + (gain if c_new == 0 else -loss) # Wait
            # If c_new == 0, we gained 'gain' in 0, lost 'loss' in 1? No.
            # If c_old == 0, c_new == 1:
            #   count0 changes by -loss (lost K222 in 0)
            #   count1 changes by +gain (gained K222 in 1)
            # If c_old == 1, c_new == 0:
            #   count1 changes by -loss
            #   count0 changes by +gain
            
            if c_old == 0:
                delta0 = -loss
                delta1 = gain
            else:
                delta0 = gain
                delta1 = -loss
            
            new_count0 = count0 + delta0
            new_count1 = count1 + delta1
            new_total = new_count0 + new_count1
            
            # Acceptance
            if new_total < best_count:
                # Always accept improvement
                count0 = new_count0
                count1 = new_count1
                best_count = new_total
                best_adj = [[row[:] for row in adj[0]], [row[:] for row in adj[1]]]
                
                if best_count == 0:
                    return best_adj
            else:
                # Revert
                adj[c_old][i][j] = 1
                adj[c_old][j][i] = 1
                adj[c_new][i][j] = 0
                adj[c_new][j][i] = 0
                
                # Simulated Annealing acceptance
                if new_total > best_count:
                    delta = new_total - best_count
                    if random.random() < pow(2.718, -delta / temp):
                        count0 = new_count0
                        count1 = new_count1
                        # Re-apply flip
                        adj[c_old][i][j] = 0
                        adj[c_old][j][i] = 0
                        adj[c_new][i][j] = 1
                        adj[c_new][j][i] = 1
                else:
                    # Equal, accept with some prob
                    if random.random() < 0.5:
                        count0 = new_count0
                        count1 = new_count1
                        adj[c_old][i][j] = 0
                        adj[c_old][j][i] = 0
                        adj[c_new][i][j] = 1
                        adj[c_new][j][i] = 1
            
            temp *= cooling
    
    # Try n=30
    adj30 = solve_for_n(30)
    if adj30 is not None:
        # Verify
        if total_k222_count(adj30, 30, 0) == 0 and total_k222_count(adj30, 30, 1) == 0:
            colors = []
            for i in range(30):
                for j in range(i + 1, 30):
                    colors.append(adj30[0][i][j])
            return {'n': 30, 'colors': colors}
    
    # Try n=29
    adj29 = solve_for_n(29)
    if adj29 is not None:
        if total_k222_count(adj29, 29, 0) == 0 and total_k222_count(adj29, 29, 1) == 0:
            colors = []
            for i in range(29):
                for j in range(i + 1, 29):
                    colors.append(adj29[0][i][j])
            return {'n': 29, 'colors': colors}
            
    # Fallback to seed
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'colors': [0 if (j - i) % 17 in squares else 1 for i in range(17) for j in range(i + 1, 17)]}

