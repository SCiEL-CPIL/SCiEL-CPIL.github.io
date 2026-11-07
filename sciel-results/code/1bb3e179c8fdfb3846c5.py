"""Program for the suite task; constructs a dense graph avoiding equal-degree path endpoints."""
# EVOLVE-BLOCK-START

def construct():
    # Complete bipartite graph K_{7,8}
    # Partition A: vertices 0..6 (7 vertices), each has degree 8
    # Partition B: vertices 7..14 (8 vertices), each has degree 7
    # Every 3-edge path s-u-v-t has s and t in opposite partitions,
    # so deg(s) != deg(t) since 7 != 8.
    a_size = 7
    partition_a = list(range(0, a_size))
    partition_b = list(range(a_size, 15))
    edges = []
    for u in partition_a:
        for v in partition_b:
            edges.append([u, v])
    return edges

# EVOLVE-BLOCK-END
