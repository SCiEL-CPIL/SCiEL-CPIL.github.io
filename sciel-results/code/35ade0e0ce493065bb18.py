<<<<<<< SEARCH
def _refine_topology(newick, labels, D, max_iterations=10):
=======
def _parse_tree_to_splits(newick, labels):
    """
    Parse Newick and return splits and node structure.
    """
    s = newick.strip()
    if s.endswith(';'):
        s = s[:-1]
    
    nodes = []
    
    def parse_subtree(idx):
        if s[idx] == '(':
            idx += 1
            children = []
            while s[idx] != ')':
                child_idx, idx = parse_subtree(idx)
                children.append(child_idx)
                if s[idx] == ',':
                    idx += 1
            idx += 1
            
            if idx < len(s) and s[idx] == ':':
                idx += 1
                while idx < len(s) and s[idx] not in ',)':
                    idx += 1
            
            node_idx = len(nodes)
            leaves = set()
            for c in children:
                leaves.update(nodes[c]['leaves'])
            nodes.append({'leaves': leaves, 'children': children})
            return node_idx, idx
        else:
            start = idx
            while idx < len(s) and s[idx] not in ',):':
                idx += 1
            label = s[start:idx]
            
            if idx < len(s) and s[idx] == ':':
                idx += 1
                while idx < len(s) and s[idx] not in ',)':
                    idx += 1
            
            node_idx = len(nodes)
            nodes.append({'leaves': {label}, 'children': []})
            return node_idx, idx
    
    root_idx, _ = parse_subtree(0)
    
    all_leaves = set(labels)
    splits = set()
    for node in nodes:
        if node['children']:
            if 0 < len(node['leaves']) < len(all_leaves):
                smaller = node['leaves'] if len(node['leaves']) < len(all_leaves) / 2 else all_leaves - node['leaves']
                splits.add(frozenset(smaller))
    
    return splits, nodes, root_idx

def _build_newick_from_nodes(nodes, root_idx):
    """Rebuild Newick string from node structure."""
    def build(idx):
        node = nodes[idx]
        if not node['children']:
            return list(node['leaves'])[0]
        else:
            child_strs = [build(c) for c in node['children']]
            return "(" + ",".join(child_strs) + ")"
    return build(root_idx) + ";"

def _refine_topology(newick, labels, D, max_iterations=10):
>>>>>>> REPLACE

Now update the `reconstruct` function to actually call the refinement:

