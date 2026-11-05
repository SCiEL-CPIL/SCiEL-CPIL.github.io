    def can_add(S, y, triples_by_vertex):
        for t in triples_by_vertex[y]:
            # t is a tuple (a,b,c)
            # check if other two are in S
            other1, other2 = (t[0], t[1]) if y != t[0] else (t[1], t[2])
            # wait, better:
            # filter t to get the two others
            others = [x for x in t if x != y]
            if others[0] in S and others[1] in S:
                return False
        return True

    def can_swap(S, y, x, triples_by_vertex):
        # S is set, y is to add, x is to remove
        # New S: S - {x} + {y}
        for t in triples_by_vertex[y]:
            others = [z for z in t if z != y]
            # Check if others are in S - {x}
            # i.e., others[0] in S and others[0] != x AND others[1] in S and others[1] != x
            if (others[0] in S and others[0] != x) and (others[1] in S and others[1] != x):
                return False
        return True
    