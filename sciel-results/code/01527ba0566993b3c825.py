import numpy as np
import itertools
import time
import sys

def construct():
      start_time = time.time()
      deadline = start_time + 140.0
      D = 11

      # 1. Generate supports
      supports = list(itertools.combinations(range(D), 4))
      N_S = len(supports) # 330
      
      # 2. Build Conflict Graph (Adjacency Matrix)
      # Conflict if intersection size is 3
      masks = [0] * N_S
      for i, s in enumerate(supports):
          m = 0
          for idx in s:
              m |= (1 << idx)
          masks[i] = m

      adj = [0] * N_S # Bitmasks for neighbors
      for i in range(N_S):
          for j in range(i + 1, N_S):
              if (masks[i] & masks[j]).bit_count() == 3:
                  adj[i] |= (1 << j)
                  adj[j] |= (1 << i)

      # 3. Solve MIS using Branch and Bound
      # We want max independent set.
      # We can use a recursive search.
      # State: current_set, candidates (nodes that can be added)
      # candidates must be independent from current_set.
      
      best_size = 0
      best_set = []
      
      # Precompute degree for ordering
      degrees = [adj[i].bit_count() for i in range(N_S)]
      
      # Greedy coloring bound for the candidate set?
      # For MIS, we color the graph induced by candidates.
      # The chromatic number is an upper bound on the clique number of the complement?
      # No.
      # For MIS in G, we can use the clique number of the complement of G induced by candidates?
      # Actually, a standard bound for MIS is the size of the maximum clique in the complement.
      # But computing max clique is hard.
      
      # Alternative: Use a simple bound.
      # If we have candidates C, the max independent set size in C is at most |C|.
      # We can improve this by checking if C is a clique in the complement? No.
      
      # Let's use a simpler heuristic: Iterated Local Search on the Support Graph.
      # It's faster to implement and likely sufficient for 330 nodes.
      
      def greedy_min_degree():
          remaining = list(range(N_S))
          s = []
          while remaining:
              min_deg = float('inf')
              min_idx = -1
              for i, v in enumerate(remaining):
                  # Count neighbors in remaining
                  # Using bitmask
                  rem_mask = 0
                  for u in remaining:
                      rem_mask |= (1 << u)
                  deg = (adj[v] & rem_mask).bit_count()
                  if deg < min_deg:
                      min_deg = deg
                      min_idx = i
              v = remaining[min_idx]
              s.append(v)
              # Remove v and its neighbors from remaining
              to_remove = adj[v] | (1 << v)
              new_remaining = []
              for u in remaining:
                  if not (to_remove >> u) & 1:
                      new_remaining.append(u)
              remaining = new_remaining
          return s

      def local_search(init_set):
          current = set(init_set)
          in_set_mask = 0
          for v in current:
              in_set_mask |= (1 << v)
          
          # Precompute blocked
          # blocked[v] = 1 if v is in current OR v has neighbor in current
          
          best = set(current)
          best_size = len(current)
          
          for _ in range(200): # Iterations
              if time.time() > deadline - 10:
                  break
              
              # Try to add a node
              # Candidates: not in current, not adjacent to current
              all_mask = (1 << N_S) - 1
              current_neighbors = 0
              for v in current:
                  current_neighbors |= adj[v]
              
              blocked = in_set_mask | current_neighbors
              candidates = all_mask & ~blocked
              
              if candidates:
                  # Pick random candidate
                  cand_list = []
                  c = candidates
                  while c:
                      lsb = c & -c
                      idx = lsb.bit_length() - 1
                      cand_list.append(idx)
                      c ^= lsb
                  v = cand_list[np.random.randint(len(cand_list))]
                  current.add(v)
                  in_set_mask |= (1 << v)
                  if len(current) > best_size:
                      best_size = len(current)
                      best = set(current)
                  continue
              
              # Stuck: Try 1-swap
              # Remove a node u, see if we can add >1 nodes
              # Or just add 1 node (size stays same, might help escape local opt)
              if current:
                  # Pick node to remove
                  u = next(iter(current))
                  current.remove(u)
                  in_set_mask &= ~(1 << u)
                  
                  # Recalculate blocked
                  current_neighbors = 0
                  for v in current:
                      current_neighbors |= adj[v]
                  blocked = in_set_mask | current_neighbors
                  candidates = all_mask & ~blocked
                  
                  if candidates:
                      cand_list = []
                      c = candidates
                      while c:
                          lsb = c & -c
                          idx = lsb.bit_length() - 1
                          cand_list.append(idx)
                          c ^= lsb
                      # Add as many as possible?
                      # Just add one random for now
                      v = cand_list[np.random.randint(len(cand_list))]
                      current.add(v)
                      in_set_mask |= (1 << v)
                      if len(current) > best_size:
                          best_size = len(current)
                          best = set(current)
                  else:
                      # Couldn't add anything, put u back
                      current.add(u)
                      in_set_mask |= (1 << u)
          
          return best

      np.random.seed(42)
      best_set = set()
      best_size = 0
      
      # Phase 1: Greedy + Local Search
      for i in range(50):
          if time.time() > deadline - 20:
              break
          init = greedy_min_degree()
          res = local_search(init)
          if len(res) > best_size:
              best_size = len(res)
              best_set = res

      # Phase 2: Simulated Annealing on Supports
      def simulated_annealing(init_set):
          current = set(init_set)
          cur_mask = 0
          for v in current:
              cur_mask |= (1 << v)
          
          best = set(current)
          best_size = len(current)
          
          T = 10.0
          for i in range(500):
              if time.time() > deadline - 5:
                  break
              
              # Move: Add, Remove, Swap
              all_mask = (1 << N_S) - 1
              current_neighbors = 0
              for v in current:
                  current_neighbors |= adj[v]
              blocked = cur_mask | current_neighbors
              add_cands = all_mask & ~blocked
              rem_cands = cur_mask
              
              r = np.random.random()
              new_set = set(current)
              
              if add_cands and r < 0.5:
                  c = add_cands
                  lsb = c & -c
                  idx = lsb.bit_length() - 1
                  new_set.add(idx)
              elif rem_cands and r < 0.75:
                  c = rem_cands
                  lsb = c & -c
                  idx = lsb.bit_length() - 1
                  new_set.discard(idx)
              elif add_cands and rem_cands:
                  c1 = add_cands
                  lsb1 = c1 & -c1
                  idx1 = lsb1.bit_length() - 1
                  c2 = rem_cands
                  lsb2 = c2 & -c2
                  idx2 = lsb2.bit_length() - 1
                  new_set.discard(idx2)
                  new_set.add(idx1)
              
              if new_set == current:
                  continue
              
              # Acceptance
              # Objective: Maximize size
              # If size increases, accept.
              # If size decreases, accept with prob exp(-delta/T)
              delta = len(new_set) - len(current)
              if delta > 0 or (T > 0.01 and np.random.random() < np.exp(delta / T)):
                  current = new_set
                  cur_mask = 0
                  for v in current:
                      cur_mask |= (1 << v)
                  if len(current) > best_size:
                      best_size = len(current)
                      best = set(current)
              
              T *= 0.95
          
          return best

      for i in range(20):
          if time.time() > deadline - 5:
              break
          init = best_set if best_set else greedy_min_degree()
          res = simulated_annealing(init)
          if len(res) > best_size:
              best_size = len(res)
              best_set = res

      # 4. Construct Vectors
      vectors = []
      for s_idx in best_set:
          support = supports[s_idx]
          for signs in itertools.product([1, -1], repeat=4):
              v = np.zeros(D, dtype=np.int8)
              for idx, sign in zip(support, signs):
                  v[idx] = sign
              vectors.append(v)

      if not vectors:
          return np.array([], dtype=np.int8).reshape(0, D)

      return np.array(vectors, dtype=np.int8)
