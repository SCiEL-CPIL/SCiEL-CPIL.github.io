def construct():
         n = 40
         set_a = list(range(20))
         set_b = list(range(20, 40))
         
         best_max_book = float('inf')
         best_edges = []
         
         # Enumerate possible R (removed edges) for the Star construction
         for R in range(0, 20):
             K = R + 1
             if K > 19: break # Can't have more intra-edges than vertices-1 in a star on 20 vertices? 
                              # Actually, star can have up to 19 edges.
             # Check if K intra-edges fit in A (size 20). Yes, max 19.
             
             removed_edges = set()
             for j in range(R):
                 removed_edges.add((0, 20 + j))
                 
             edges = []
             max_book = 0
             
             # Add bipartite
             for u in set_a:
                 for v in set_b:
                     if (u, v) not in removed_edges:
                         edges.append([u, v])
                         
             # Add intra
             for i in range(1, K + 1):
                 edges.append([0, i])
                 
             # Calculate max book size
             # Intra (0, i): 20 - R
             # Bip (0, v) not removed: K
             # Bip (u, v) u!=0: 0
             
             book_intra = 20 - R
             book_bip = K
             current_max = max(book_intra, book_bip)
             
             if current_max < best_max_book:
                 best_max_book = current_max
                 best_edges = edges
                 
         return best_edges
