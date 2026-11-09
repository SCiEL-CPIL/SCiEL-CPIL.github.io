"""Combined: complementary deletion (fewest-conflicts) + vectorized local search + kicks + targeted polish."""
import numpy as np
import time
import random

# EVOLVE-BLOCK-START

def construct():
    d = 7
    n_vertices = 1 << d  # 128
    n_edges = n_vertices * d // 2  # 448

    # Build edge list and mapping
    edge_to_idx = {}
    idx_to_edge = []
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(idx_to_edge)
                idx_to_edge.append((u, v))
                edge_to_idx[(u, v)] = idx

    # Precompute faces: each face is a set of 4 edge indices
    faces = []
    face_edges = [[] for _ in range(n_edges)]

    for i in range(d):
        for j in range(i + 1, d):
            mask_i = 1 << i
            mask_j = 1 << j
            for x in range(n_vertices):
                if (x & mask_i) or (x & mask_j):
                    continue
                v0 = x
                v1 = x ^ mask_i
                v2 = x ^ mask_j
                v3 = x ^ mask_i ^ mask_j

                e01 = (min(v0, v1), max(v0, v1))
                e02 = (min(v0, v2), max(v0, v2))
                e13 = (min(v1, v3), max(v1, v3))
                e23 = (min(v2, v3), max(v2, v3))

                idx01 = edge_to_idx[e01]
                idx02 = edge_to_idx[e02]
                idx13 = edge_to_idx[e13]
                idx23 = edge_to_idx[e23]

                face_idx = len(faces)
                faces.append([idx01, idx02, idx13, idx23])

                for ei in [idx01, idx02, idx13, idx23]:
                    face_edges[ei].append(face_idx)

    num_faces = len(faces)
    faces_arr = np.array(faces, dtype=np.int32)  # shape (num_faces, 4)

    # Precompute for each edge, the 3 other edges in each face it belongs to
    edge_face_others = [[] for _ in range(n_edges)]
    for fidx, face in enumerate(faces):
        for pos, e in enumerate(face):
            others = [face[k] for k in range(4) if k != pos]
            edge_face_others[e].append((fidx, others))

    # Vectorized "others" arrays per edge
    edge_others_flat = []
    for e in range(n_edges):
        flist = []
        for fidx, others in edge_face_others[e]:
            flist.extend(others)
        edge_others_flat.append(np.array(flist, dtype=np.int32) if flist else np.array([], dtype=np.int32))

    deadline = time.time() + 115

    def can_add(idx, sel):
        """Check if edge idx can be added without creating a C4 (vectorized)."""
        others = edge_others_flat[idx]
        if len(others) == 0:
            return True
        n_f = len(others) // 3
        others_reshaped = others.reshape(n_f, 3)
        sel_others = sel[others_reshaped]
        return not np.any(np.all(sel_others, axis=1))

    def can_swap(idx_rm, idx_add, sel):
        """Check if removing idx_rm and adding idx_add keeps C4-free (vectorized)."""
        others = edge_others_flat[idx_add]
        if len(others) == 0:
            return True
        n_f = len(others) // 3
        others_reshaped = others.reshape(n_f, 3)
        sel_others = sel[others_reshaped].astype(np.int8)
        rm_mask = (others_reshaped == idx_rm).astype(np.int8)
        effective = sel_others * (1 - rm_mask)
        return not np.any(np.all(effective == 1, axis=1))

    def count_edges(sel):
        return int(np.sum(sel))

    def greedy_fill(sel, order=None, max_time=None):
        """Add back edges that can be added without creating C4."""
        if order is None:
            order = list(range(n_edges))
        for idx in order:
            if max_time is not None and time.time() > max_time:
                break
            if not sel[idx]:
                if can_add(idx, sel):
                    sel[idx] = True

    def local_search_1flip(sel, rng, max_iters=200):
        """Try to add random missing edges."""
        improved = True
        while improved and max_iters > 0:
            improved = False
            max_iters -= 1
            for _ in range(80):
                idx = rng.randint(0, n_edges - 1)
                if sel[idx]:
                    continue
                if can_add(idx, sel):
                    sel[idx] = True
                    improved = True
                    break
        return sel

    def local_search_2swap(sel, rng, max_iters=150):
        """Try 2-swap: remove one edge, add another."""
        improved = True
        while improved and max_iters > 0:
            improved = False
            max_iters -= 1
            for _ in range(40):
                idx_rm = rng.randint(0, n_edges - 1)
                if not sel[idx_rm]:
                    continue
                idx_add = rng.randint(0, n_edges - 1)
                if sel[idx_add]:
                    continue
                if can_swap(idx_rm, idx_add, sel):
                    sel[idx_rm] = False
                    sel[idx_add] = True
                    improved = True
                    break
        return sel

    def local_search_3flip(sel, rng, max_iters=100):
        """Try 3-flip: remove two edges, add one. Net -1 but can unlock future +1 moves."""
        improved = False
        for _ in range(max_iters):
            if time.time() > deadline - 3:
                break
            idx_rm1 = rng.randint(0, n_edges - 1)
            idx_rm2 = rng.randint(0, n_edges - 1)
            if not sel[idx_rm1] or not sel[idx_rm2] or idx_rm1 == idx_rm2:
                continue
            idx_add = rng.randint(0, n_edges - 1)
            if sel[idx_add]:
                continue
            others = edge_others_flat[idx_add]
            if len(others) == 0:
                sel[idx_rm1] = False
                sel[idx_rm2] = False
                sel[idx_add] = True
                improved = True
                break
            n_f = len(others) // 3
            others_reshaped = others.reshape(n_f, 3)
            sel_others = sel[others_reshaped].astype(np.int8)
            rm_mask = ((others_reshaped == idx_rm1) | (others_reshaped == idx_rm2)).astype(np.int8)
            effective = sel_others * (1 - rm_mask)
            if not np.any(np.all(effective == 1, axis=1)):
                sel[idx_rm1] = False
                sel[idx_rm2] = False
                sel[idx_add] = True
                improved = True
                break
        return sel

    def complementary_deletion(rng):
        """Start with all edges, remove to break all C4s (fewest-conflicts heuristic)."""
        sel = np.ones(n_edges, dtype=bool)

        face_order = list(range(num_faces))
        rng.shuffle(face_order)

        for fidx in face_order:
            if time.time() > deadline:
                break
            face = faces[fidx]
            # Check if all 4 edges are still selected
            if not (sel[face[0]] and sel[face[1]] and sel[face[2]] and sel[face[3]]):
                continue
            # Remove the edge with the FEWEST other still-violating faces
            best_to_remove = None
            best_score = None
            for e in face:
                if not sel[e]:
                    continue
                conflict_count = 0
                for ofidx, others in edge_face_others[e]:
                    if ofidx == fidx:
                        continue
                    if all(sel[oe] for oe in others):
                        conflict_count += 1
                if best_score is None or conflict_count < best_score:
                    best_score = conflict_count
                    best_to_remove = e
            if best_to_remove is not None:
                sel[best_to_remove] = False

        return sel

    def full_local_search(sel, rng):
        """Iterative 1-flip and 2-swap until no improvement."""
        prev_count = -1
        for _ in range(20):
            if time.time() > deadline - 2:
                break
            cnt = count_edges(sel)
            if cnt == prev_count:
                break
            prev_count = cnt

            sel = local_search_1flip(sel, rng, max_iters=100)
            if time.time() > deadline - 2:
                break
            sel = local_search_2swap(sel, rng, max_iters=80)
            if time.time() > deadline - 2:
                break
            if rng.random() < 0.3:
                sel = local_search_3flip(sel, rng, max_iters=50)

        return sel

    def polish_pass(sel, rng):
        """Final polish: deterministic adds + targeted blocking-face swaps."""
        for idx in range(n_edges):
            if time.time() > deadline - 1:
                break
            if not sel[idx] and can_add(idx, sel):
                sel[idx] = True

        missing = [i for i in range(n_edges) if not sel[i]]
        rng.shuffle(missing)
        for idx_add in missing[:300]:
            if time.time() > deadline - 1:
                break
            if sel[idx_add]:
                continue
            others = edge_others_flat[idx_add]
            if len(others) == 0:
                sel[idx_add] = True
                continue
            n_f = len(others) // 3
            others_reshaped = others.reshape(n_f, 3)
            sel_others = sel[others_reshaped]
            blocking = np.where(np.all(sel_others, axis=1))[0]
            if len(blocking) == 0:
                sel[idx_add] = True
                continue
            for bf in blocking:
                if time.time() > deadline - 1:
                    break
                face_others = others_reshaped[bf]
                for e in face_others:
                    if sel[e] and can_swap(e, idx_add, sel):
                        sel[e] = False
                        sel[idx_add] = True
                        break
                if sel[idx_add]:
                    break

        for idx in range(n_edges):
            if time.time() > deadline - 1:
                break
            if not sel[idx] and can_add(idx, sel):
                sel[idx] = True

        return sel

    best_sel = None
    best_count = 0

    # Deterministic greedy order (reused every restart so restarts differ only by
    # deletion order and kicks, not by random fill order).
    base_order = list(range(n_edges))

    restart = 0
    while time.time() < deadline - 6:
        restart += 1
        rng = random.Random(1000 + restart * 7919)

        # Phase 1: Complementary deletion (randomized face order)
        sel = complementary_deletion(rng)

        # Phase 2: Deterministic greedy fill
        if time.time() < deadline - 8:
            greedy_fill(sel, order=base_order, max_time=deadline - 8)

        # Phase 3: Full local search
        if time.time() < deadline - 6:
            sel = full_local_search(sel, rng)

        cnt = count_edges(sel)
        if cnt > best_count:
            best_count = cnt
            best_sel = sel.copy()

        # Kick-based restart from the best: perturb, re-fill, re-search.
        if best_count >= 300 and time.time() < deadline - 10:
            sel2 = best_sel.copy()
            # Kick: remove a handful of random edges to escape the local optimum
            for _ in range(rng.randint(2, 6)):
                k = rng.randint(0, n_edges - 1)
                sel2[k] = False
            greedy_fill(sel2, order=base_order, max_time=deadline - 10)
            if time.time() < deadline - 8:
                sel2 = full_local_search(sel2, rng)
            cnt2 = count_edges(sel2)
            if cnt2 > best_count:
                best_count = cnt2
                best_sel = sel2

    # Convert to edge list
    if best_sel is None:
        return []

    # Final polish pass on the best solution
    if time.time() < deadline - 2:
        rng_polish = random.Random(12345)
        best_sel = polish_pass(best_sel, rng_polish)

    edges = []
    for i in range(n_edges):
        if best_sel[i]:
            u, v = idx_to_edge[i]
            edges.append([u, v])

    return edges

# EVOLVE-BLOCK-END
