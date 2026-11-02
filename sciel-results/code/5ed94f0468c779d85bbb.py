    candidates = []
    
    # Generate compact candidates for N=11
    # Shape 1: 5-4-2 triangle
    axial_c1 = [
        (0,0), (1,0), (2,0), (3,0), (4,0),
        (0,1), (1,1), (2,1), (3,1),
        (0,2), (1,2)
    ]
    candidates.append(axial_c1)
    
    # Shape 2: 4-4-3 triangle
    axial_c2 = [
        (0,0), (1,0), (2,0), (3,0),
        (0,1), (1,1), (2,1), (3,1),
        (0,2), (1,2), (2,2)
    ]
    candidates.append(axial_c2)
    
    # Shape 3: 3-4-4 triangle
    axial_c3 = [
        (0,0), (1,0), (2,0),
        (0,1), (1,1), (2,1), (3,1),
        (0,2), (1,2), (2,2), (3,2)
    ]
    candidates.append(axial_c3)
    
    # Shape 4: 2-4-5 triangle
    axial_c4 = [
        (0,0), (1,0),
        (0,1), (1,1), (2,1), (3,1),
        (0,2), (1,2), (2,2), (3,2), (4,2)
    ]
    candidates.append(axial_c4)
    
    # Shape 5: 1-4-6? No, 1+4+6=11.
    # Row 0: 1
    # Row 1: 4
    # Row 2: 6
    axial_c5 = [
        (0,0),
        (0,1), (1,1), (2,1), (3,1),
        (0,2), (1,2), (2,2), (3,2), (4,2), (5,2)
    ]
    candidates.append(axial_c5)
    
    # Shape 6: 6-4-1
    axial_c6 = [
        (0,0), (1,0), (2,0), (3,0), (4,0), (5,0),
        (0,1), (1,1), (2,1), (3,1),
        (0,2)
    ]
    candidates.append(axial_c6)
    
    # Shape 7: 3-5-3
    axial_c7 = [
        (0,0), (1,0), (2,0),
        (0,1), (1,1), (2,1), (3,1), (4,1),
        (0,2), (1,2), (2,2)
    ]
    candidates.append(axial_c7)
    
    # Shape 8: 4-3-4
    axial_c8 = [
        (0,0), (1,0), (2,0), (3,0),
        (0,1), (1,1), (2,1),
        (0,2), (1,2), (2,2), (3,2)
    ]
    candidates.append(axial_c8)
    
    # Shape 9: 2-5-4
    axial_c9 = [
        (0,0), (1,0),
        (0,1), (1,1), (2,1), (3,1), (4,1),
        (0,2), (1,2), (2,2), (3,2)
    ]
    candidates.append(axial_c9)
    
    # Shape 10: 4-5-2
    axial_c10 = [
        (0,0), (1,0), (2,0), (3,0),
        (0,1), (1,1), (2,1), (3,1), (4,1),
        (0,2), (1,2)
    ]
    candidates.append(axial_c10)
    
    # Shape 11: 5-3-3
    axial_c11 = [
        (0,0), (1,0), (2,0), (3,0), (4,0),
        (0,1), (1,1), (2,1),
        (0,2), (1,2), (2,2)
    ]
    candidates.append(axial_c11)
    
    # Shape 12: 3-3-5
    axial_c12 = [
        (0,0), (1,0), (2,0),
        (0,1), (1,1), (2,1),
        (0,2), (1,2), (2,2), (3,2), (4,2)
    ]
    candidates.append(axial_c12)
