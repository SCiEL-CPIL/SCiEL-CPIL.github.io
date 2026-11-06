def get_sorted_triples_with_d(a, b, c, d):
    # a<b<c
    # Insert d
    verts = [a, b, c, d]
    verts.sort()
    return (verts[0], verts[1], verts[2]), (verts[0], verts[1], verts[3]), (verts[0], verts[2], verts[3]), (verts[1], verts[2], verts[3])
    # Wait, we need (a,b,d), (a,c,d), (b,c,d).
    # If we have sorted verts v0,v1,v2,v3.
    # The triples are (v0,v1,v2), (v0,v1,v3), (v0,v2,v3), (v1,v2,v3).
    # We know (a,b,c) is one of them.
    # We need the other 3.
