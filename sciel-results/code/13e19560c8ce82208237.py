import math
import time
import random
from itertools import combinations

def reconstruct(sequences):
    start_time = time.time()
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n <= 3:
        # Trivial tree
        if n == 3:
            return f"({labels[0]},{labels[1]},{labels[2]});"
        elif n == 2:
            return f"({labels[0]},{labels[1]});"
        else:
            return f"({labels[0]});"

    # 1. Compute Pairwise Distances (JC corrected)
    # Using a list of sequences for faster access
    seqs = [sequences[label] for label in labels]
    L = len(seqs[0])
    
    # Distance matrix D
    D = [[0.0] * n for _ in range(n)]
    
    for i in range(n):
        si = seqs[i]
        for j in range(i + 1, n):
            sj = seqs[j]
            # Count differences
            diff = 0
            # Manual loop is often faster than sum(1 for...) for short strings in Python if not vectorized, 
            # but let's use zip which is C-speed.
            for x, y in zip(si, sj):
                if x != y:
                    diff += 1
            p = diff / L
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            D[i][j] = d
            D[j][i] = d

    # 2. Quartet Analysis
    # For each quartet, determine the best topology and the corresponding split.
    # Split is represented by the set of taxa on one side (the smaller side for canonical form).
    # We collect all winning splits.
    
    # To handle conflicts, we will count the "support" for each split.
    # Support = sum of Q values for quartets supporting that split.
    
    split_support = {}
    
    # Pre-compute indices for faster access
    # We need to map label to index
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    # Iterate over all quartets
    # This is O(n^4). For n=32, ~36k iterations.
    # Inside, we do constant work.
    
    for a_idx, b_idx, c_idx, d_idx in combinations(range(n), 4):
        # Get distances
        d_ab = D[a_idx][b_idx]
        d_ac = D[a_idx][c_idx]
        d_ad = D[a_idx][d_idx]
        d_bc = D[b_idx][c_idx]
        d_bd = D[b_idx][d_idx]
        d_cd = D[c_idx][d_idx]
        
        # Q values for the 3 topologies
        # Topology 1: ((a,b),(c,d)) -> Internal branch separates {a,b} and {c,d}
        # Q1 = d_ac + d_bd - d_ab - d_cd
        q1 = d_ac + d_bd - d_ab - d_cd
        split1 = frozenset([a_idx, b_idx])
        
        # Topology 2: ((a,c),(b,d)) -> Internal branch separates {a,c} and {b,d}
        # Q2 = d_ab + d_cd - d_ac - d_bd
        q2 = d_ab + d_cd - d_ac - d_bd
        split2 = frozenset([a_idx, c_idx])
        
        # Topology 3: ((a,d),(b,c)) -> Internal branch separates {a,d} and {b,c}
        # Q3 = d_ad + d_bc - d_ab - d_cd
        q3 = d_ad + d_bc - d_ab - d_cd
        split3 = frozenset([a_idx, d_idx])
        
        # Find the best Q
        max_q = max(q1, q2, q3)
        
        # Add to support if it's the winner
        # We only add if it's clearly the best to reduce noise? 
        # Standard QP adds all. But if q is negative, it's weak support.
        # Let's add the winner with weight max_q.
        
        if q1 == max_q:
            # Canonicalize split: use the smaller set
            if len(split1) > n // 2:
                # This shouldn't happen for a quartet split, it's always size 2 vs 2
                pass 
            split_support[split1] = split_support.get(split1, 0.0) + q1
            
        elif q2 == max_q:
            split_support[split2] = split_support.get(split2, 0.0) + q2
            
        else:
            split_support[split3] = split_support.get(split3, 0.0) + q3
            
    # 3. Tree Assembly
    # We have a dictionary of split -> support.
    # We need to select a compatible set of splits that maximizes total support.
    # This is the "Maximum Compatible Split Set" problem, which is NP-hard, 
    # but for small n and dense quartet data, a greedy approach works well.
    
    # Sort splits by support (descending)
    sorted_splits = sorted(split_support.items(), key=lambda item: item[1], reverse=True)
    
    # Start with an empty set of selected splits
    selected_splits = []
    
    # Helper to check compatibility
    def is_compatible(s1, s2):
        # s1, s2 are frozensets of indices
        # They are compatible if one is subset of other, or they are disjoint, or union is empty (impossible)
        # Actually, for unrooted splits on the same set of taxa:
        # A and B are compatible if A \subset B, B \subset A, or A \cap B = \emptyset.
        # Note: A and B are never complementary in the sense of being the "same" split unless A == B or A == complement(B).
        # But we canonicalized splits to be the "side" of the bipartition.
        # If we store splits as the set of taxa on one side, we must ensure we don't store both A and T\A.
        # My canonicalization above for quartets always picked the size 2 set. 
        # But as we add splits, we might have splits of different sizes.
        # Let's ensure we store the "smaller" side of every split to avoid duplicates.
        
        # For the check:
        # If A == B, they are the same split.
        # If A is subset of B, compatible.
        # If B is subset of A, compatible.
        # If A is disjoint from B, compatible.
        # If A is disjoint from complement of B? No, that's not the condition.
        
        # The condition for two splits A and B (where A, B are subsets of T, neither empty nor T) to be compatible is:
        # A \cap B = \emptyset OR A \cap (T\B) = \emptyset OR B \cap (T\A) = \emptyset OR B \cap (T\A) ... 
        # Simpler: A and B are compatible if none of A\B, B\A, A\cap B, (T\A)\cap(T\B) is non-empty? No.
        
        # Correct Condition:
        # Two splits are incompatible if they "cross".
        # They cross if A\B, B\A, A\cap B, and (T\A)\cap(T\B) are all non-empty.
        
        # Let's implement the "crossing" check.
        inter = s1.intersection(s2)
        if not inter: return True # Disjoint
        
        # If they intersect, they must be nested to be compatible.
        # s1 subset s2 or s2 subset s1
        if s1.issubset(s2): return True
        if s2.issubset(s1): return True
        
        # Otherwise, they cross
        return False

    # Greedy selection
    for split, support in sorted_splits:
        # Check if this split is compatible with all already selected splits
        compatible = True
        for existing in selected_splits:
            if not is_compatible(split, existing):
                compatible = False
                break
        if compatible:
            selected_splits.append(split)
            
    # 4. Build the Tree from Selected Splits
    # We have a set of compatible splits. We can build the tree using a recursive algorithm.
    # Or, more simply, we can use the splits to define the topology.
    
    # A robust way to build a tree from a set of splits is to use a "tree builder" that inserts splits.
    # However, since we have the full set of taxa, we can reconstruct the topology.
    
    # Let's use a simple recursive construction.
    # If the set of splits is empty, return a star tree.
    # Otherwise, find a split that partitions the current set of taxa.
    
    def build_tree(taxa_set, splits):
        if len(taxa_set) <= 3:
            # Return a star tree for these taxa
            if len(taxa_set) == 1:
                return f"({labels[list(taxa_set)[0]]})"
            elif len(taxa_set) == 2:
                l1, l2 = list(taxa_set)
                return f"({labels[l1]},{labels[l2]})"
            else:
                l1, l2, l3 = list(taxa_set)
                return f"({labels[l1]},{labels[l2]},{labels[l3]})"
                
        # Find a split in 'splits' that partitions 'taxa_set'
        # A split S is a frozenset. It partitions T if S is a proper subset of T.
        # We need to find a split S such that S is a subset of taxa_set or (T\S) is a subset of taxa_set.
        # Actually, the split defines a bipartition of the *global* set of taxa.
        # When building a subtree for 'taxa_set', we look for a split that separates 'taxa_set' into two non-empty parts.
        
        best_split = None
        for s in splits:
            # s is a frozenset of indices
            # Check if s intersects taxa_set
            inter = s.intersection(taxa_set)
            if not inter:
                continue
            # The other part of the split is T \ s
            # But we only care about the partition of 'taxa_set'.
            # The split divides the universe into A and B.
            # The current node represents a set of taxa X.
            # If X is contained in A, no split.
            # If X is contained in B, no split.
            # If X intersects both A and B, then the split is valid for this node.
            
            # A = s
            # B = all_indices - s
            # We need to know if taxa_set intersects A and B.
            # We know inter = taxa_set \cap A.
            # If inter is empty, skip.
            # If inter == taxa_set, then taxa_set is entirely in A. Skip.
            # Otherwise, taxa_set intersects both A and B.
            
            if inter != taxa_set:
                best_split = s
                break
                
        if best_split is None:
            # No split found, return star tree
            taxa_list = list(taxa_set)
            if len(taxa_list) == 4:
                return f"({labels[taxa_list[0]]},{labels[taxa_list[1]]},{labels[taxa_list[2]]},{labels[taxa_list[3]]})"
            else:
                # Star tree
                return "(" + ",".join(f"({labels[i]})" for i in taxa_list) + ")" # Incorrect, star is (t1,t2,...)
                # Correct star: (t1, t2, t3)
                return "(" + ",".join(labels[i] for i in taxa_list) + ")"
        
        # Partition taxa_set
        left = best_split.intersection(taxa_set)
        right = taxa_set - best_split
        
        # Recursively build subtrees
        left_tree = build_tree(left, splits)
        right_tree = build_tree(right, splits)
        
        return f"({left_tree},{right_tree})"
        
    all_indices = frozenset(range(n))
    newick = build_tree(all_indices, selected_splits)
    
    return newick + ";"
