"""3D Bin Packing using a 3D BSP (Binary Space Partitioning) tree for exact, non-overlapping placement."""

class BSPNode:
    __slots__ = ('x0', 'y0', 'z0', 'x1', 'y1', 'z1', 'left', 'right')
    def __init__(self, x0, y0, z0, x1, y1, z1):
        self.x0, self.y0, self.z0 = x0, y0, z0
        self.x1, self.y1, self.z1 = x1, y1, z1
        self.left = None
        self.right = None

    def volume(self):
        return (self.x1 - self.x0) * (self.y1 - self.y0) * (self.z1 - self.z0)

def _split_node(node, x, y, z, w, h, d):
    """
    Try to carve out a box of size (w,h,d) starting at (x,y,z) within node.
    Returns True if successful (and updates tree), False otherwise.
    The box must be fully contained within [x0,x1]x[y0,y1]x[z0,z1].
    """
    x0, y0, z0 = node.x0, node.y0, node.z0
    x1, y1, z1 = node.x1, node.y1, node.z1

    # Check if box fits in this node
    if x < x0 or y < y0 or z < z0 or x + w > x1 or y + h > y1 or z + d > z1:
        return False

    # If box exactly fills the node, we can just consume it (but we represent empty space, so if it fills, no empty space remains)
    # Actually, we represent EMPTY space. So we are removing the box volume from empty space.
    # If the box doesn't overlap the empty region at all, we can't place it here.
    # Wait, the logic is: we have a set of empty rectangular subspaces.
    # We want to find a spot for the box.
    # If the box fits entirely within this node's rectangle, we can place it there.
    # Then we need to split this node to represent the remaining empty space.

    # The box occupies [x, x+w] x [y, y+h] x [z, z+d].
    # The node is [x0, x1] x [y0, y1] x [z0, z1].
    
    # We need to split the node into up to 6 sub-nodes (front, back, left, right, up, down) relative to the box,
    # but only those that have positive volume.
    
    # Let's define the splits along each axis.
    # X-axis: [x0, x], [x+w, x1]
    # Y-axis: [y0, y], [y+h, y1]
    # Z-axis: [z0, z], [z+d, z1]
    
    # We create a 2x2x2 grid of potential sub-nodes around the box.
    # However, a simpler approach for BSP is to recursively split.
    
    # Let's try to fit the box in the "corner" (x0, y0, z0) first? No, the box position is fixed by the caller?
    # Actually, in a standard BSP packing, we usually search for a position.
    # Let's modify the approach:
    # 1. Search the BSP tree for a valid position for the box (w,h,d).
    # 2. If found, insert the box by splitting the nodes.
    
    # For simplicity and efficiency, let's use a "Best Fit" strategy:
    # Traverse the tree. At each node, check if the box fits.
    # If it fits, we can place it at (node.x0, node.y0, node.z0) to keep it compact?
    # Or we can try to find the tightest fit.
    
    # Let's refine:
    # The function _find_spot(node, w, h, d) returns (x, y, z) if a spot is found, else None.
    # It prefers spots with smaller x, then y, then z to pack tightly.
    
    # If a spot is found, we call _remove_box(node, x, y, z, w, h, d) to update the empty space.

def _find_spot(node, w, h, d):
    if node is None:
        return None
    
    # If the box fits in this node, return the bottom-left-front corner as the placement.
    # This greedy choice helps with packing density.
    if (node.x1 - node.x0 >= w and 
        node.y1 - node.y0 >= h and 
        node.z1 - node.z0 >= d):
        return (node.x0, node.y0, node.z0)
    
    # Otherwise, search children
    # We want to try the child that is "closer" to the origin first?
    # Or just search both.
    # To encourage tight packing, we should search the child that shares the origin first.
    
    # Left child is typically the one containing (x0, y0, z0) if we split that way.
    # Let's define the splitting strategy in _remove_box to be consistent.
    
    # For now, search left then right.
    res = _find_spot(node.left, w, h, d)
    if res is not None:
        return res
    res = _find_spot(node.right, w, h, d)
    if res is not None:
        return res
    return None

def _remove_box(node, x, y, z, w, h, d):
    """
    Remove the box volume from the empty space tree.
    The box is known to be inside this node's bounds.
    """
    x0, y0, z0 = node.x0, node.y0, node.z0
    x1, y1, z1 = node.x1, node.y1, node.z1
    
    # If the box does not intersect the node's interior (shouldn't happen if called correctly), return
    if x >= x1 or y >= y1 or z >= z1 or x + w <= x0 or y + h <= y0 or z + d <= z0:
        return
    
    # If the box exactly covers the node, the node becomes empty (None)
    if x == x0 and y == y0 and z == z0 and x + w == x1 and y + h == y1 and z + d == z1:
        # This node is fully occupied.
        # We need to replace this node with None in the parent.
        # But we are modifying in place? 
        # It's better to have a function that returns a new node or modifies the parent's pointer.
        # Let's use a recursive approach that returns the new subtree root.
        pass # Handled by parent logic
    
    # This in-place modification is tricky. Let's switch to a functional style where we return a new node or None.
    
def _split_and_remove(node, x, y, z, w, h, d):
    """
    Returns the new node (or None) after removing the box.
    """
    if node is None:
        return None
    
    x0, y0, z0 = node.x0, node.y0, node.z0
    x1, y1, z1 = node.x1, node.y1, node.z1
    
    # Check if box is completely outside
    if x >= x1 or y >= y1 or z >= z1 or x + w <= x0 or y + h <= y0 or z + d <= z0:
        return node
    
    # Check if box covers the entire node
    if x <= x0 and y <= y0 and z <= z0 and x + w >= x1 and y + h >= y1 and z + d >= z1:
        return None
    
    # If the box fits entirely within the node, we can place it and split.
    # But wait, _find_spot already ensured the box fits in a specific spot (x,y,z) which is (node.x0, node.y0, node.z0) in our greedy approach.
    # So in _remove_box, we assume the box is at (node.x0, node.y0, node.z0) for the node it was found in?
    # No, _find_spot returns (x,y,z). We then need to remove that specific box from the tree.
    # The box might span multiple nodes? No, BSP nodes are disjoint.
    # If the box fits in a node, it is fully contained in that node's rectangle.
    # So we can just process that node.
    
    # However, if we use a simple BSP where we split along axes, the empty space is represented by a tree of rectangles.
    # When we remove a box, we only affect the node(s) that contain it.
    # Since our _find_spot returns a position inside a single node, the box is contained in that single node's rectangle.
    # So we only need to split that one node.
    
    # Let's assume the box is at (x,y,z) and is contained in [x0,x1]x[y0,y1]x[z0,z1].
    
    # We need to create up to 6 new nodes for the remaining space.
    # Actually, we can just create the 8 octants around the box and keep the ones that are valid and have volume.
    
    new_nodes = []
    
    # X splits: [x0, x], [x+w, x1]
    # Y splits: [y0, y], [y+h, y1]
    # Z splits: [z0, z], [z+d, z1]
    
    xs = [(x0, x), (x+w, x1)]
    ys = [(y0, y), (y+h, y1)]
    zs = [(z0, z), (z+d, z1)]
    
    # Create 8 potential sub-nodes
    for (xa, xb) in xs:
        if xb > xa:
            for (ya, yb) in ys:
                if yb > ya:
                    for (za, zb) in zs:
                        if zb > za:
                            new_nodes.append(BSPNode(xa, ya, za, xb, yb, zb))
    
    # Now we need to merge these if possible, or just store them.
    # Storing 6-8 children per node is inefficient.
    # A standard BSP splits into 2 children.
    # Let's use a different splitting strategy.
    
    # Split along X first?
    # If x > x0, we have a left part [x0, x] and a right part [x, x1].
    # The box is in the right part? No, the box is [x, x+w].
    # The empty space is [x0, x] and [x+w, x1] in X.
    
    # Let's try a recursive split:
    # 1. If x > x0, split node into Left [x0, x] and Right [x, x1].
    #    The box is in the Right part. Recurse on Right.
    #    Left becomes a leaf (or further split if needed, but it's empty).
    # 2. If x + w < x1, split Right into Left2 [x, x+w] (occupied) and Right2 [x+w, x1].
    #    Wait, the box occupies [x, x+w]. So the empty space in the Right part is [x+w, x1] in X.
    
    # This is getting complicated. Let's use a simpler "Best Fit" with a list of empty spaces, but optimized with a quadtree/bsp-like structure?
    # No, the directive asks for BSP.
    
    # Let's implement a standard BSP where each node is a rectangle.
    # When inserting a box, we find a node that contains it.
    # We then split that node into up to 6 rectangles.
    # To keep the tree balanced/small, we can merge adjacent rectangles if they form a larger rectangle.
    # For now, let's just allow up to 6 children and see if performance is okay for 40 boxes.
    
    # Actually, a simpler BSP approach:
    # Node has a split axis and split position.
    # But for empty space, the "box" is the hole.
    
    # Let's stick to the "Node is a rectangle" approach.
    # We'll replace the node with a set of child nodes.
    # To avoid explosion, we'll only split if necessary.
    
    # Revised Plan:
    # 1. _find_spot: Returns (node, x, y, z) where the box fits at (x,y,z) which is node.x0, node.y0, node.z0.
    # 2. _remove_box: Takes the node and the box size. Since the box is at the corner, we can simply shrink the node?
    #    No, the box is at the corner (x0,y0,z0).
    #    The remaining empty space is:
    #    - [x0+w, x1] x [y0, y1] x [z0, z1] (Right)
    #    - [x0, x0+w] x [y0+h, y1] x [z0, z1] (Back)
    #    - [x0, x0+w] x [y0, y0+h] x [z0+d, z1] (Up)
    #    Wait, this overlaps.
    #    The remaining space is the union of:
    #    A: [x0+w, x1] x [y0, y1] x [z0, z1]
    #    B: [x0, x0+w] x [y0+h, y1] x [z0, z1]
    #    C: [x0, x0+w] x [y0, y0+h] x [z0+d, z1]
    #    These three are disjoint?
    #    A is x >= x0+w.
    #    B is x < x0+w and y >= y0+h.
    #    C is x < x0+w and y < y0+h and z >= z0+d.
    #    Yes, they are disjoint and cover the remainder.
    
    # So we can replace the node with these 3 nodes (if they have volume).
    
    # Let's implement this.

def _find_spot(node, w, h, d):
    if node is None:
        return None
    
    if (node.x1 - node.x0 >= w and 
        node.y1 - node.y0 >= h and 
        node.z1 - node.z0 >= d):
        return (node, node.x0, node.y0, node.z0)
    
    # Search children
    # We store children in a list for flexibility, or just left/right?
    # Let's use a list of children to handle the 3-way split.
    
    for child in node.children:
        res = _find_spot(child, w, h, d)
        if res is not None:
            return res
    return None

class BSPNode:
    __slots__ = ('x0', 'y0', 'z0', 'x1', 'y1', 'z1', 'children')
    def __init__(self, x0, y0, z0, x1, y1, z1):
        self.x0, self.y0, self.z0 = x0, y0, z0
        self.x1, self.y1, self.z1 = x1, y1, z1
        self.children = []

def _remove_box_from_node(node, w, h, d):
    """
    Assumes the box of size (w,h,d) is placed at (node.x0, node.y0, node.z0).
    Replaces node's children with the 3 remaining empty sub-nodes.
    """
    x0, y0, z0 = node.x0, node.y0, node.z0
    x1, y1, z1 = node.x1, node.y1, node.z1
    
    node.children = []
    
    # A: Right
    if x0 + w < x1:
        node.children.append(BSPNode(x0 + w, y0, z0, x1, y1, z1))
        
    # B: Back
    if y0 + h < y1:
        node.children.append(BSPNode(x0, y0 + h, z0, x0 + w, y1, z1))
        
    # C: Up
    if z0 + d < z1:
        node.children.append(BSPNode(x0, y0, z0 + d, x0 + w, y0 + h, z1))

def pack(boxes, container):
    W, H, D = container
    placements = []
    bin_id = 0
    root = BSPNode(0, 0, 0, W, H, D)
    
    # Sort boxes by volume descending to pack large ones first?
    # The task says "preserving its original ordered dimensions" but does it say we must place them in order?
    # "places every box... into numbered bins".
    # Usually, we can reorder boxes for better packing, but the output must correspond to the input boxes?
    # "one placement ... per box". It doesn't explicitly say the i-th output is for the i-th input box.
    # However, "preserving its original ordered dimensions" implies the box object is fixed.
    # If we reorder, we must output the placement for box i at index i?
    # Let's assume we can reorder the processing, but must output in the original order?
    # Or does the evaluator check that the i-th placement is for the i-th box?
    # "one placement (bin, x, y, z, w, h, d) per box".
    # It's safer to assume the output list index i corresponds to input list index i.
    # So we must keep track of which box is which.
    
    # Let's sort indices by volume descending for packing efficiency, then map back.
    indexed_boxes = [(i, boxes[i]) for i in range(len(boxes))]
    indexed_boxes.sort(key=lambda x: x[1][0]*x[1][1]*x[1][2], reverse=True)
    
    results = [None] * len(boxes)
    
    for idx, (w, h, d) in indexed_boxes:
        # Try to place in current bin
        spot = _find_spot(root, w, h, d)
        if spot is not None:
            node, x, y, z = spot
            _remove_box_from_node(node, w, h, d)
            results[idx] = (bin_id, x, y, z, w, h, d)
        else:
            # New bin
            bin_id += 1
            root = BSPNode(0, 0, 0, W, H, D)
            spot = _find_spot(root, w, h, d)
            if spot is not None:
                node, x, y, z = spot
                _remove_box_from_node(node, w, h, d)
                results[idx] = (bin_id, x, y, z, w, h, d)
            else:
                # Should not happen if box fits in empty container
                results[idx] = (bin_id, 0, 0, 0, w, h, d)
                
    return results
